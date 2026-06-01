import numpy as np
import os
import time
import shutil
import psutil
import glob
import pandas as pd
import socket
import platform
import pickle
import re
from pathlib import Path

try:                   # Ensure Blosc uses only LSF-allocated cores, not machine total
    import numcodecs as _nc
    _nc.blosc.set_nthreads(int(os.environ.get('LSB_DJOB_NUMPROC', 1)))
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, warp
from struct import unpack, pack
from tqdm.notebook import tqdm
from collections import defaultdict
import mrcfile
import cv2
cv2.setNumThreads(int(os.environ.get('LSB_DJOB_NUMPROC', 1)))
try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff

from dask.distributed import Client
from dask.distributed import as_completed
from IPython.display import IFrame, display
from ClusterWrap.clusters import janelia_lsf_cluster

from scipy.signal import savgol_filter
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates

from sklearn.linear_model import (LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
    RidgeCV,
    LassoCV)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from FIBSEM_gs_py.FIBSEM_gs import (FIBSEM_frame,
                        ShiftTransform,
                        RotationShiftTransform,
                        XScaleShiftTransform,
                        ScaleShiftTransform,
                        RegularizedAffineTransform,
                        get_min_max_thresholds,
                        extract_keypoints_descr_files,
                        extract_image_intensity, 
                        determine_transformations_files,
                        convert_tr_matr_into_deformation_field,
                        Perform_2D_fit,
                        flatten_image_fast,
                        evaluate_FIBSEM_frames_dataset)

from FIBSEM_gs_py.FIBSEM_help_functions_gs import (check_DASK,
                                                    find_FWHM,
                                                    dask_remove_file,
                                                    elapsed_since,
                                                    get_process_memory,
                                                    format_bytes,
                                                    read_kwargs_xlsx,
                                                    parse_metadata_file,
                                                    read_image_coordinates)



def build_weight_array(shape, **kwargs):
    '''
    Builds a 2D array of weights for image blending. gleb.shtengel@gmail.com 11.2025
    Parameters:
    -----------
    shape : tuple (y, x)
        Shape of the array to create the weights.
    
    kwargs:
    ----------
    weight_min : np.float32
        weight_min for weight. Default is 1
    weight_max : np.float32
        weight_max for weight. Default is 512

    Returns:
    ----------
    weights
    '''
    weight_min = kwargs.get('weight_min', 1.0)
    weight_max = kwargs.get('weight_max', 512.0)
    indy, indx = np.indices(shape).astype(np.float32)
    indx_r = np.flip(indx)
    indy_r = np.flip(indy)
    weights = np.clip((np.min(np.array([indx, indx_r, indy, indy_r]), axis=0) + weight_min), weight_min, weight_max)
    return weights


def split_translation_int_fract(transformation_matrix):
    '''
    Split transformation_matrix into large integer shift and fractional transformation.

    Parameters:
        transformation_matrix : (3, 3) array — affine transformation matrix
    Returns:
        transformation_matrix_fract : (3, 3) float32 — matrix with fractional translations only
        dx, dy : int — integer translation components extracted from the matrix
    '''
    dx, dy = -np.round(transformation_matrix[0:2, 2]).astype(np.int32)
    delta_matrix = np.array([[0, 0, dx],
                         [0, 0, dy],
                         [0, 0,  0]])
    return transformation_matrix + delta_matrix, dx, dy


def combine_deformation_fields(DF1, DF2, interpolation=cv2.INTER_LINEAR):
    """
    Combine two deformation fields: DFjoint = DF2 ∘ DF1
    i.e. first apply DF1, then DF2.

    Parameters:
        DF1: (YResolution, XResolution - left_crop, 2) float32 — local nonlinear warp, absolute coords in src_tile
        DF2: (YResolution, XResolution - left_crop, 2) float32 — local shift (small enough to not violate CV2.remap SHRT_MAX limitation).
        interpolation : int
            Interpolation type to be used. default is cv2.INTER_LINEAR

    Returns:
        DFjoint:   (YResolution, XResolution - left_crop, 2) float32 — composed deformation field
    """
    DF1_x = DF1[..., 0].astype(np.float32)
    DF1_y = DF1[..., 1].astype(np.float32)

    DF2_x = DF2[..., 0].astype(np.float32)
    DF2_y = DF2[..., 1].astype(np.float32)

    DFjoint_x = cv2.remap(DF2_x, DF1_x, DF1_y, interpolation, borderMode=cv2.BORDER_CONSTANT,    borderValue=np.nan)
    DFjoint_y = cv2.remap(DF2_y, DF1_x, DF1_y, interpolation, borderMode=cv2.BORDER_CONSTANT,    borderValue=np.nan)

    return np.stack([DFjoint_x, DFjoint_y], axis=-1)


def _add_warped_to_mosaic(tile, xi, yi, mosaic, mosaic_weight, **kwargs):
    """
    Warp and accumulate a tile into the mosaic using weighted blending.

    Parameters:
        tile:   2D array
        xi, yi — integer mosaic placement coords [x, y]
        mosaic:   output array to write into
        mosaic_weight :  total mosaic weight for tile merging

    kwargs:
    weight_min : float
        vmin for weight. Default is 1.
    weight_max : float
        vmax for weight. Default is 512
    """

    weight_min = kwargs.get('weight_min', 1.0)
    weight_max = kwargs.get('weight_max', 512.0)
    
    mosaic_sy, mosaic_sx = mosaic.shape[:2]
    tile_sy, tile_sx = tile.shape[:2]

    xa = xi + tile_sx
    ya = yi + tile_sy

    # Clamp to mosaic bounds
    cxi = max(xi, 0)
    cyi = max(yi, 0)
    cxa = min(xa, mosaic_sx)
    cya = min(ya, mosaic_sy)

    if cxi >= cxa or cyi >= cya:
        return  # tile falls entirely outside mosaic
        
    # Corresponding region in warped tile
    txi = cxi - xi
    tyi = cyi - yi
    txa = txi + (cxa - cxi)
    tya = tyi + (cya - cyi)

    tile_out = tile[tyi:tya, txi:txa]
    tile_weight = build_weight_array(tile_out.shape, weight_min = weight_min, weight_max = weight_max)

    nan_mask = np.isnan(tile_out)
    tile_weight[nan_mask] = 0
    tile_out = np.nan_to_num(tile_out, copy=True, nan=0.0)

    mosaic[cyi:cya, cxi:cxa] += tile_out*tile_weight
    mosaic_weight[cyi:cya, cxi:cxa] += tile_weight


def _write_zarr3_shard_s0_from_tiles(params, deformation_field, **kwargs):
    """
    DASK worker — build one s0 shard of a zarr v3 store from tile data.
    Composites contributing tiles into a shard-local (sz, sy, sx) ZYX buffer
    via `_add_warped_to_mosaic`, optionally transposes to the output axis
    order, then writes the whole shard atomically.

    params : list
        [output_zarr_path,
         x0, y0, z0,                       # canvas-coords shard origin (XYZ)
         sx, sy, sz,                       # shard size at s0 (XYZ); edge shards may clip
         layer_ids,                        # list of layer indices in [z0, z0+sz)
         tiles_per_layer,                  # list, per layer: [(tile_id, fl, tr_matr_single, I0, scale), ...]
         image_name, fill_value,
         weight_min, weight_max,
         left_crop,
         flatten_kwargs,                   # dict or None  — passed through to optional flattening
         axis_perm,                        # (0,1,2) | (2,1,0) etc — source ZYX → output
         dtp,
         U8_range,                         # None or [umin, umax]
         verbose]

    deformation_field : 3D array | np.nan
        Shared by all workers; pass via DASK_client.scatter(..., broadcast=True).
    """
    import os, time
    import numpy as np
    import zarr

    t0 = time.time()
    (output_zarr_path,
     x0, y0, z0,
     sx, sy, sz,
     layer_ids,
     tiles_per_layer,
     image_name, fill_value,
     weight_min, weight_max,
     left_crop,
     flatten_kwargs,
     axis_perm,
     dtp,
     U8_range,
     verbose) = params

    kwargs_tt = {
        'verbose': False,
        'interpolation': kwargs.get('interpolation'),
        'border_value':  kwargs.get('border_value', np.nan),
        'border_mode':   kwargs.get('border_mode'),
    }
    kwargs_awp = {'weight_min': weight_min, 'weight_max': weight_max}

    # Allocate ZYX shard buffer.
    shard_buf_zyx = np.empty((sz, sy, sx), dtype=dtp)

    for layer_idx, (layer_id, tile_list) in enumerate(zip(layer_ids, tiles_per_layer)):
        # Shard-local mosaic + weights for this layer.
        layer_mosaic   = np.zeros((sy, sx), dtype=np.float32)
        layer_weights  = np.zeros((sy, sx), dtype=np.float32)

        for (j, fl, tr_matr_single, tile_I0, tile_scale) in tile_list:
            tile_params = [j, fl, image_name, tr_matr_single,
                           sy, sx,                # NOT used by transform_tile for output sizing
                           left_crop, tile_I0, tile_scale]
            tile_warped, xi, yi = transform_tile(tile_params, deformation_field, **kwargs_tt)
            # xi, yi are the tile's placement in CANVAS coords.
            # Convert to shard-local coords and let _add_warped_to_mosaic clip:
            _add_warped_to_mosaic(
                tile_warped, xi - x0, yi - y0,
                layer_mosaic, layer_weights,
                **kwargs_awp,
            )

        # Normalise (same idiom as assemble_layer).
        np.clip(layer_weights, weight_min, weight_max * max(len(tile_list), 1),
                out=layer_weights)
        np.divide(layer_mosaic, layer_weights, out=layer_mosaic)
        np.nan_to_num(layer_mosaic, nan=fill_value, copy=False)

        # Optional per-layer flattening (same as assemble_layer's flatten path).
        if flatten_kwargs is not None:
            offset = flatten_kwargs.get('mosaic_Scaling_offset', 0.0)
            layer_mosaic = flatten_image_fast(
                layer_mosaic - offset,
                flatten_kwargs['mosaic_correction_intercept'],
                flatten_kwargs['mosaic_correction_coeffs'],
                flatten_kwargs['mosaic_correction_degree'],
                flatten_kwargs['mosaic_correction_bins'],
            ) + offset

        # Cast to dtp.
        if dtp == np.uint8 and U8_range is not None:
            U8_min, U8_max = float(U8_range[0]), float(U8_range[1])
            scale = 255.0 / max(U8_max - U8_min, 1e-6)
            np.subtract(layer_mosaic, U8_min, out=layer_mosaic)
            np.multiply(layer_mosaic, scale,  out=layer_mosaic)
            np.clip(layer_mosaic, 0, 255, out=layer_mosaic)
            shard_buf_zyx[layer_idx] = layer_mosaic.astype(np.uint8)
        else:
            shard_buf_zyx[layer_idx] = layer_mosaic.astype(dtp)

    # Transpose to output axis order. ZYX = (0,1,2); XYZ = (2,1,0).
    if axis_perm != (0, 1, 2):
        shard_out = shard_buf_zyx.transpose(axis_perm)
    else:
        shard_out = shard_buf_zyx

    # Destination slices in output axis order.
    src_slices_zyx = (slice(z0, z0 + sz), slice(y0, y0 + sy), slice(x0, x0 + sx))
    dst_slices = tuple(src_slices_zyx[axis_perm[i]] for i in range(3))

    arr = zarr.open(output_zarr_path, mode='r+')['s0']
    arr[dst_slices] = shard_out

    elapsed = time.time() - t0
    return {
        'shard_origin_xyz': (x0, y0, z0),
        'shard_size_xyz':   (sx, sy, sz),
        'nbytes': shard_out.nbytes,
        'elapsed_s': elapsed,
    }


def _downsample_zarr3_shard(params):
    """
    DASK worker — downsample one shard from level (lvl-1) to level lvl
    within the same zarr v3 store. Mirrors the per-shard pattern used by
    convert_ome_zarr_v2_to_v3 (one shard = one task = one atomic write).

    params : list
        [output_zarr_path, src_lvl_arr_path, dst_lvl_arr_path,
         src_slices, dst_slices, downsample_factor, dtp]
    """
    import time
    import numpy as np
    import zarr

    t0 = time.time()
    (output_zarr_path, src_arr_path, dst_arr_path,
     src_slices, dst_slices, downsample_factor, dtp) = params

    grp = zarr.open_group(str(output_zarr_path), mode='r+')
    src = grp[src_arr_path]
    dst = grp[dst_arr_path]

    data = np.asarray(src[src_slices])

    # Trim to a multiple of downsample_factor along every axis, then average downsample_factor-blocks.
    trimmed = tuple((data.shape[d] // downsample_factor) * downsample_factor for d in range(data.ndim))
    if any(t == 0 for t in trimmed):
        return {'arr_path': dst_arr_path, 'elapsed_s': time.time() - t0,
                'note': 'empty after trim'}
    cropped = data[:trimmed[0], :trimmed[1], :trimmed[2]]
    new_shape = tuple(s // downsample_factor for s in trimmed)
    downsampled = (
        cropped
        .reshape(new_shape[0], downsample_factor, new_shape[1], downsample_factor, new_shape[2], downsample_factor)
        .mean(axis=(1, 3, 5))
        .astype(dtp)
    )

    # Write whole shard atomically; only fill the actually-downsampled region
    # (edge shards may be smaller than dst_slices' nominal extent).
    out_slices = tuple(slice(dst_slices[d].start,
                             dst_slices[d].start + downsampled.shape[d])
                       for d in range(data.ndim))
    dst[out_slices] = downsampled

    return {
        'arr_path': dst_arr_path,
        'dst_slices': str(dst_slices),
        'nbytes': downsampled.nbytes,
        'elapsed_s': time.time() - t0,
    }


def transform_tile(tile_params, deformation_field, **kwargs):
    '''
    Transforms individual tile to add to the montage. gleb.shtengel@gmail.com 11.2025
    Assumes following order of the tile transformations: left_crop ->  nonlinear transformation defined by deformation_field -> transformation determined by tr_matr_single
    1. Apply left_crop.
    2. Split tr_matr_single into large integer shift and fractional transformation.
    3. Combine non-linear deformation defined by deformation_field with fractional transformation.
    4. Perform the transformation determined by a combined field calculated in previous step.
    5. Return the transformed image and coordinates for placing the transformed tile into the mosaic.

    Parameters:
    -----------
    tile_params : list :  j, fl, image_name, tr_matr_single, montage_ysz, montage_xsz, left_crop, I0, scale
        j : int, tile ID
        fl : str, filename for the tile
        image_name : str, image name ('RawImageA' or 'RawImageB')
        tr_matr_single : 3x3 array : transformation matrix (backward map: mosaic → corrected+cropped tile)
        montage_xsz : int : montage x-size in pixels
        montage_ysz : int : montage y-size in pixels
        left_crop : int : number of pixels to crop from the left of the deformed tile
        I0 : np.float32 : intensity offset for tile normalization
        scale : np.float32 : intensity scale for tile normalization

    deformation_field : 3D array, shape (YResolution, XResolution - left_crop, 2)
        Deformation field for distortion correction.
        Pass np.nan (scalar) to skip distortion correction (registration only).

    kwargs:
    -----------
    verbose : bool
        If True, the intermediate results are displayed. Default is False.
    interpolation : int
        Default is cv2.INTER_LINEAR
    border_value : float
        borderValue for cv2.remap. Default is np.nan
    border_mode : int
        borderMode for cv2.remap. Default is cv2.BORDER_CONSTANT

    Returns:
    ----------
    tile_transformed, dx, dy

    '''
    j, fl, image_name, tr_matr_single, montage_ysz, montage_xsz, left_crop, I0, scale = tile_params
    verbose = kwargs.get('verbose', False)
    interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
    border_value = kwargs.get('border_value', np.nan)
    border_mode = kwargs.get('border_mode', cv2.BORDER_CONSTANT)

    fr = FIBSEM_frame(fl)
    
    # Step 1. Apply left_crop
    if image_name == 'RawImageB':
        tile_initial = fr.RawImageB.astype(np.float32)[:, left_crop:]
    else:
        tile_initial = fr.RawImageA.astype(np.float32)[:, left_crop:]
    tile_initial_rescaled = (tile_initial - I0) * scale + I0

    # Step 2. Split tr_matr_single into large integer shift (dx, dy) and fractional transformation (transformation_matrix_fract).
    transformation_matrix_fract, dx, dy = split_translation_int_fract(tr_matr_single)

    # Step 3. Combine non-linear deformation defined by deformation_field with fractional transformation.
    perform_deformation = not np.all(np.isnan(deformation_field))
    if perform_deformation:
        df_tr_matr_fract = convert_tr_matr_into_deformation_field(transformation_matrix_fract, (fr.YResolution, fr.XResolution-left_crop)).astype(np.float32)
        df_joint = combine_deformation_fields(deformation_field, df_tr_matr_fract, interpolation=interpolation)
    else:
        # No distortion correction — apply registration transform only
        df_joint = convert_tr_matr_into_deformation_field(transformation_matrix_fract, (fr.YResolution, fr.XResolution-left_crop)).astype(np.float32)

    # Step 4 — Single remap: one interpolation on image data. Perform the transformation determined by a combined field calculated in previous step.
    tile_transformed = cv2.remap(
        tile_initial_rescaled,
        df_joint[..., 0],
        df_joint[..., 1],
        interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )

    if verbose:
        print('cv2.remap returned image with shape:    ', tile_transformed.shape)
        print('Transformed tile will be placed with offsets dx={:d},  dy={:d}'.format(dx, dy))

    return tile_transformed, dx, dy
    

def overlay_montage_grid(ax, montage_object, **kwargs):
    '''
    Overlays grid of tile boundaries over the montage image. gleb.shtengel@gmail.com 11.2025
    
    Parameters:
    -----------
    ax : matplotlib axis object
    montage_object : montage object
    
    kwargs:
    -----------
    linestyle : string
        Matplotlib linestyle. Default is 'dashed'.
    linewidth : np.float32
        Matplotlib linewidth. Default is 1.0.
    color : string
        Matplotlib color. Default 'cyan'.
    tile_positions_actual : bool
        If True (default), use actual solved positions from tr_matr. If False, use nominal positions from montage_object.FirstPixels.
    tile_positions : 2D array or list
        Actual tile positions. Default is derived from montage_object.tr_matr (layer 0 positions).
    left_crop : int 
        Cropping value for cropping the image from the left side (used along with deformation_field or on its own). Default is 0 - no cropping.
    dx : int
        X-size of the tile. Default is montage_object.XResolution-left_crop
    dy : int
        Y-size of the tile. Default is montage_object.YResolution
    bin_factor : int
        If the displayed mosaic was binned by this factor before being passed
        to imshow, divide all overlay coordinates (xi, yi, dx, dy) by this
        factor so the grid aligns with the displayed image. Default is 1
        (no binning).
    '''
    linestyle = kwargs.get('linestyle', 'dashed')
    linewidth = kwargs.get('linewidth', 1.0)
    color = kwargs.get('color', 'cyan')
    left_crop = kwargs.get('left_crop', 0)
    dx = kwargs.get('dx', montage_object.XResolution-left_crop)
    dy = kwargs.get('dy', montage_object.YResolution)
    tile_positions_actual = kwargs.get('tile_positions_actual', True)
    bin_factor = kwargs.get('bin_factor', 1)
    if not isinstance(bin_factor, int) or bin_factor < 1:
        raise ValueError(
            f"overlay_montage_grid: bin_factor must be a positive int (got {bin_factor!r})."
        )

    if tile_positions_actual:
        # Default: derive (positive) tile positions from tr_matr on the fly.
        # Callers can override by passing tile_positions= explicitly.
        tile_positions = kwargs.get('tile_positions', -montage_object.tr_matr[0, :, 0:2, 2])
        for tile_position in tile_positions:
            xi, yi = tile_position
            rect_patch = patches.Rectangle(
                (xi / bin_factor, yi / bin_factor),
                (dx - 2) / bin_factor, (dy - 2) / bin_factor,
                linewidth=linewidth, linestyle=linestyle,
                edgecolor=color, facecolor='none')
            ax.add_patch(rect_patch)
    else:
        X0 = montage_object.FirstPixels[0, 0]
        Y0 = montage_object.FirstPixels[0, 1]
        for FirstPixel_pair in montage_object.FirstPixels:
            xi = np.max((FirstPixel_pair[0]- X0, 0))
            yi = np.max((FirstPixel_pair[1]- Y0, 0))
            dx_loc = dx  + np.min((FirstPixel_pair[0]- X0, 0))
            dy_loc = dy  + np.min((FirstPixel_pair[1]- Y0, 0))
            rect_patch = patches.Rectangle(
                (xi / bin_factor, yi / bin_factor),
                (dx_loc - 2) / bin_factor, (dy_loc - 2) / bin_factor,
                linewidth=linewidth, linestyle=linestyle,
                edgecolor=color, facecolor='none')
            ax.add_patch(rect_patch)


def remap_tile(img, deformation_field, **kwargs):
    '''
    Remap Image using CV2.remap (using deformation field). gleb.shtengel@gmail.com 11.2025
    remap_tile is needed to work around CV2.remap SHRT_MAX limitation (CV2.remap cannot work with images larger than 32767).
    1. The deformation field is shifted - constant shifts (shift_x and shift_y) are subtracted so that the output array image has as few empty pixels as possible.
    2. Then the image is deformed and returned along with shifts, indicating where this tile needs to be placed in the mosaic.
    
    Parameters:
    ---------
        img : 2D array
            input image
        deformation_field : 3D array
            deformation field. Last index has dimension 2. deformation_field[:, :, 0] is x-deformation, deformation_field[:, :, 1] is y-deformation.
            
    kwargs:
    ---------
        interpolation : int
            interpolation used by CV2.remap. Default is cv2.INTER_LINEAR (==1).
        borderValue : int
            borderValue used by CV2.remap. Default is np.nan.
        verbose : bool
            If True, print out intermediate resulys. Default is False.
    
    Returns:
    ---------- 
    image_deformed, shift_x, shift_y
    '''
    interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
    borderValue = kwargs.get('borderValue', np.nan)
    verbose = kwargs.get('verbose', False)
    
    img_shape = img.shape
    shift_x = int(np.min((np.nanmin(deformation_field[:, :, 0]), 0))) # Find the leftmost source coordinate in the entire field — but only if it is negative.
    shift_y = int(np.min((np.nanmin(deformation_field[:, :, 1]), 0))) # Find the top-most source coordinate in the entire field — but only if it is negative.

    if verbose:
        print('shift_x={:d},  shift_y={:d}'.format(shift_x, shift_y))
    df_shifted = deformation_field.astype(np.float32) - np.array([shift_x, shift_y], dtype=np.float32)  # Shift the deformation field to make all coordinates non-negative.
    # Compute the expanded canvas size
    # The canvas must be at least as large as the original image AND large enough to contain every source coordinate the shifted field can address.
    dfx_min = np.nanmin(df_shifted[:, :, 0])  # Return minimum of an array ignoring any NaNs
    dfx_max = np.nanmax(df_shifted[:, :, 0])
    dfy_min = np.nanmin(df_shifted[:, :, 1])
    dfy_max = np.nanmax(df_shifted[:, :, 1])
    xsz_new = np.max((img_shape[1], int(dfx_max - dfx_min + 1)))
    ysz_new = np.max((img_shape[0], int(dfy_max - dfy_min + 1)))
    if verbose:
        print('dfx_min={:.2f},  dfx_max={:.2f}'.format(dfx_min, dfx_max))
        print('dfy_min={:.2f},  dfy_max={:.2f}'.format(dfy_min, dfy_max))
        print('Original Image Shape: ', img_shape)
        print('Deformed Image Shape: ', (ysz_new, xsz_new))
    # Embed the tile in the expanded canvas 
    image_expanded = np.zeros((ysz_new, xsz_new), dtype=np.float32)
    image_expanded[0:img_shape[0],0:img_shape[1]] = img
    # Embed the shifted deformation field in the same expanded grid
    df_expanded = np.zeros((ysz_new, xsz_new, 2), dtype=np.float32)
    df_expanded[0:img_shape[0],0:img_shape[1], :] = df_shifted
                           
    image_deformed = cv2.remap(image_expanded,
                               df_expanded[:, :, 0],
                               df_expanded[:, :, 1], interpolation=interpolation, borderValue=borderValue)
    # shift_x and shift_y are returned so transform_tile knows where to place the output in the mosaic
    return image_deformed, shift_x, shift_y


def find_Transform_ECC(img1, img2, **kwargs):
    '''
    Find Transformation using cv2.findTransformECC on parts of images with approximately known and small image overlaps.
    Works much faster than if performed on whole images. gleb.shtengel@gmail.com 11.2025.
    
    
    Parameters:
    ----------
    img1 : 2D array
    img2 : 2D array
    
    ----------
    kwargs:
    image_margins : tuple of 2 ints
        Parts of images to be used. It is assumed that img1 is to the left and above of the img2.
        Subsets img1[-ymargin:, -xmargin:] and  img2[0:ymargin, 0:xmargin] will be used for correlation.
        Default is full images, so image_margins = (ymargin, xmargin) = img1.shape
    warp_matrix : 3x2 initial guess of the transformation matrix.
        Default is np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    motion : target transformation.
        Default is cv2.MOTION_TRANSLATION
    criteria : criteria.
        Default is (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    repeats : int
        repeat internally this many times. Default is 2.
    verbose : boolean
            Display intermediate results. Default is False.
            
    Returns:
    ----------
    warp_matrix, error_code
        warp_matrix : Updated warp matrix. If failed, returns original warp_matrix.
        error_code : CV2.error code. 0 if no error.
    '''
    ysz, xsz = img1.shape
    ymargin, xmargin =  kwargs.get('image_margins', (ysz, xsz))
    warp_matrix = kwargs.get('warp_matrix', np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    motion = kwargs.get('motion', cv2.MOTION_TRANSLATION)
    criteria = kwargs.get('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7))
    repeats = kwargs.get('repeats', 2)
    verbose = kwargs.get('verbose', False)
    
    dx = xsz - xmargin
    dy = ysz - ymargin
    matr_shift = np.array(((0, 0, dx), (0, 0, dy)), dtype=np.float32)
    warp_matrix = warp_matrix + matr_shift
    error_code = 0
    try:
        for ii in np.arange(repeats):
            (cc, warp_matrix) = cv2.findTransformECC(img1[-ymargin:, -xmargin:], img2[0:ymargin, 0:xmargin], warp_matrix, motion, criteria)
        tx = warp_matrix[0, 2]
        ty = warp_matrix[1, 2]
        if verbose:
            print('Estimated translation: tx={:.3f}, ty={:.3f}'.format((tx - dx), (ty - dy)))
    except cv2.error as e:
        error_code = e
        if verbose:
            print('ECC failed to converge: ', error_code)
    return warp_matrix - matr_shift, error_code


def find_Transform_ECC_DASK(params, deformation_field):
    '''
    Find Transformation using cv2.findTransformECC on parts of images with approximately known and small image overlaps.
    deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
    If no deformation is needed, still pass a deformation field, but it will not be used (kwarg['perform_deformation']=False)
    Works much faster than if performed on whole images. gleb.shtengel@gmail.com 11.2025.
    
    Parameters:
    ----------
    params : list of [fname1, fname2, kwargs]
    deformation_field : 3D array
        Array with dimensions (YResolution, XResolution - left_crop, 2). Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
    
    ----------
    kwargs:
    interpolation : int
        Interpolation type as defined in CV2. Default is cv2.INTER_LINEAR.
    fill_value : float
        Fill value for outside pixels in cv2.remap. Default is 0.0.
    image_margins : tuple of 2 ints
        Parts of images to be used. It is assumed that img1 is to the left and above of the img2.
        Subsets img1[-ymargin:, -xmargin:] and  img2[0:ymargin, 0:xmargin] will be used for correlation.
        Default is full images, so image_margins = (ymargin, xmargin) = img1.shape
    left_crop : int 
        Cropping value for cropping the image from the left side (used along with deformation_field or on its own). Default is 0 - no cropping.
    warp_matrix : 3x2 initial guess of the transformation matrix.
        Default is np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    motion : target transformation.
        Default is cv2.MOTION_TRANSLATION
    criteria : criteria.
        Default is (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    repeats : int
        repeat internally this many times. Default is 2.
    verbose : boolean
        Display intermediate results. Default is False.
            
    Returns:
    ----------
    warp_matrix, error_code
        warp_matrix : Updated warp matrix. If failed, returns original warp_matrix.
        error_code : CV2.error code. 0 if no error.
    '''
    fname1, fname2, kwargs = params
    ftype = kwargs.get('ftype', 0)
    perform_deformation = not np.all(np.isnan(deformation_field))
    interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
    fill_value = kwargs.get('fill_value', 0)
    left_crop = kwargs.get('left_crop', 0)
    verbose = kwargs.get('verbose', False)
    if perform_deformation:
        if verbose:
            print('find_Transform_ECC_DASK: performing deformation and cropping, left_crop={:d}'.format(left_crop))
        img1 = cv2.remap(FIBSEM_frame(fname1, ftype=ftype).RawImageA_8bit_thresholds()[0].astype(np.float32)[:, left_crop:], deformation_field[:, :, 0].astype(np.float32), deformation_field[:, :, 1].astype(np.float32), interpolation=interpolation, borderValue=fill_value).astype(np.uint8)
        img2 = cv2.remap(FIBSEM_frame(fname2, ftype=ftype).RawImageA_8bit_thresholds()[0].astype(np.float32)[:, left_crop:], deformation_field[:, :, 0].astype(np.float32), deformation_field[:, :, 1].astype(np.float32), interpolation=interpolation, borderValue=fill_value).astype(np.uint8)
    else:
        if verbose:
            print('find_Transform_ECC_DASK: no deformation, left_crop={:d}'.format(left_crop))
        img1 = FIBSEM_frame(fname1, ftype=ftype).RawImageA_8bit_thresholds()[0][:, left_crop:]
        img2 = FIBSEM_frame(fname2, ftype=ftype).RawImageA_8bit_thresholds()[0][:, left_crop:]
    ymargin, xmargin =  kwargs.get('image_margins', img1.shape)
    kwargs['image_margins'] = (ymargin, xmargin-left_crop)
    warp_matrix, error_code = find_Transform_ECC(img1, img2, **kwargs)

    return warp_matrix, error_code


def assemble_layer(params, deformation_field, **kwargs):
    '''
    Assembles layer. Worker function called by assemble_layer_mosaic and save_stack. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    params = [layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max,
          fill_value, Xsize, Ysize, left_crop, tile_I0s, tile_scales,
          return_layer_array, save_tif, tif_fname, dtp, bin_factor, verbose]
        layer_id : int
            Layer ID should be a value between -1 and self.nz_tiles-1. -1 means the last layer will be assembled.
        fls_layer : list
            List of files for individual tiles.
        image_name : str
            Image name ('RawImageA' or 'RawImageB').
        tr_matr_layer : list
            List of transformation matrices for individual tiles.
        weight_min : np.float32
            vmin for weight.
        weight_max : np.float32
            vmax for weight.
        fill_value : int
            The value to assign to pixels outside the transformed image bounds.
        Xsize : int
            Overall Mosaic width (pixels).
        Ysize : int
            Overall Mosaic height (pixels).
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_field or on its own).
        return_layer_array : bool
            If True, the layer mosaic array is returned to the caller. If False, np.zeros(1) is returned.
        save_tif : bool
            If True, the layer mosaic is saved into tif file
        tif_fname : str
            path for the TIF file
        dtp : data type
        bin_factor : int
            Output binning factor (>= 1). If > 1, the assembled mosaic is binned
            by mean over (bin_factor x bin_factor) blocks before saving / returning.
            Output shape becomes (Ysize // bin_factor, (Xsize - left_crop) // bin_factor).
            Trailing edge pixels that don't form a complete bin are dropped.
            Default is 1 (no binning).
        verbose : boolean
            Display intermediate results.
    
    deformation_field : 3D array
        Array with dimensions (YResolution, XResolution - left_crop, 2). Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.

    kwargs:
    -----------
    verbose : bool
        If True, the intermediate results are displayed. Default is False.
    interpolation : int
        Default is cv2.INTER_LINEAR
    border_value : float
        borderValue for cv2.remap. Default is np.nan
    border_mode : int
        borderMode for cv2.remap. Default is cv2.BORDER_CONSTANT
    local_DASK_client : client
        local DASK client, may be used if called by assemble_layer_mosaic
    DASK_client_retries : int
        Number of DASK_client_retries. Default is 3.
    flatten_mosaic : bool
        If True, apply mosaic-level field flattening. Default is False.
    mosaic_correction_intercept : float
        Intercept for this image_name's polynomial fit.
    mosaic_correction_coeffs : 1D array
        Coefficients for this image_name's polynomial fit.
    mosaic_correction_degree : int
        Polynomial degree.
    mosaic_correction_bins : int
        Binning factor used during fitting.
    mosaic_Scaling_offset : float
        Scaling offset for Raw images (e.g. Scaling[1,0]). Only used when flatten_mosaic=True.
        Set to 0.0 for ImageA/ImageB sources. Default is 0.0.
    dtp : data type
    U8_range : list [U8_min, U8_max]
        Optional conversion range for uint8 output. Only used when dtp=np.uint8.
        Data is clipped to [U8_min, U8_max] and rescaled to [0, 255] before casting.
        Default is None (plain cast, data must already be in [0, 255]).

    Returns:
    ----------
    layer_mosaic, layer_id
    '''
    verbose = kwargs.get('verbose', False)
    interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
    border_value = kwargs.get('border_value', np.nan)
    border_mode = kwargs.get('border_mode', cv2.BORDER_CONSTANT)
    local_DASK_client = kwargs.get('local_DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    flatten_mosaic = kwargs.get('flatten_mosaic', False)
    mosaic_correction_intercept = kwargs.get('mosaic_correction_intercept', None)
    mosaic_correction_coeffs = kwargs.get('mosaic_correction_coeffs', None)
    mosaic_correction_degree = kwargs.get('mosaic_correction_degree', 2)
    mosaic_correction_bins = kwargs.get('mosaic_correction_bins', 10)
    mosaic_Scaling_offset = kwargs.get('mosaic_Scaling_offset', 0.0)

    use_DASK, status = check_DASK(local_DASK_client, verbose = False)

    layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max, fill_value, \
    Xsize, Ysize, left_crop, tile_I0s, tile_scales, return_layer_array, save_tif, tif_fname, \
    dtp, bin_factor, verbose = params
    layer_mosaic = np.zeros((Ysize, Xsize-left_crop), dtype=np.float32)
    layer_mosaic_weights = np.zeros((Ysize, Xsize-left_crop), dtype=np.float32)
    tile_params_mult = []
    for fl, (j, tr_matr_single) in zip(tqdm(fls_layer, desc = 'Building tile parameter sets', display = verbose), enumerate(tr_matr_layer)):
        tile_params_mult.append([j, fl, image_name, tr_matr_single, Ysize, Xsize, left_crop, tile_I0s[j], tile_scales[j]])

    kwargs_tt = {'verbose' : verbose,
                'interpolation' : interpolation,
                'border_value' : border_value,
                'border_mode' : border_mode}

    kwargs_awp = {'weight_min' : weight_min,
                'weight_max' : weight_max}

    # transform_tile(tile_params, deformation_field, **kwargs)
    # tile_params : list :  j, fl, image_name, tr_matr_single, montage_ysz, montage_xsz, left_crop, I0, scale
    # Returns: tile_transformed, dx, dy

    if len(tile_params_mult)>0:
        if use_DASK:
            shared_data_future = local_DASK_client.scatter(deformation_field, broadcast=True)
            futures = local_DASK_client.map(transform_tile, tile_params_mult, deformation_field = shared_data_future, retries = DASK_client_retries, **kwargs_tt)
            for future in tqdm(as_completed(futures), total=len(futures), desc='Assembling ' + image_name + ' mosaic layer'):
                    tile_out, xi, yi = future.result()
                    _add_warped_to_mosaic(tile_out, xi, yi, layer_mosaic, layer_mosaic_weights, **kwargs_awp)
        else:
            for tile_params in tqdm(tile_params_mult, desc = 'Building mosaic for layer_id={:d}'.format(layer_id), display = verbose):
                if verbose:
                    print('Performing transform_tile with the following parameters:')
                    print(tile_params)
                tile_out, xi, yi = transform_tile(tile_params, deformation_field, **kwargs_tt)
                if verbose:
                    print('Output is:')
                    print('tile_out.shape=', tile_out.shape)
                    print('xi={:d}, yi={:d}'.format(xi, yi))
                _add_warped_to_mosaic(tile_out, xi, yi, layer_mosaic, layer_mosaic_weights, **kwargs_awp)
        
        np.clip(layer_mosaic_weights, weight_min, weight_max*len(fls_layer), out=layer_mosaic_weights)
        np.divide(layer_mosaic, layer_mosaic_weights, out=layer_mosaic)
        del layer_mosaic_weights
        np.nan_to_num(layer_mosaic, nan=fill_value, copy=False)
        if flatten_mosaic and mosaic_correction_coeffs is not None:
            # Subtract offset, flatten, re-add offset
            layer_mosaic = flatten_image_fast(
                layer_mosaic - mosaic_Scaling_offset,
                mosaic_correction_intercept,
                mosaic_correction_coeffs,
                mosaic_correction_degree,
                mosaic_correction_bins) + mosaic_Scaling_offset

        # Optional output binning. Done in float32 BEFORE the dtype cast so we
        # don't lose precision in the mean. Trailing edge pixels that don't
        # form a complete bin are dropped.
        if bin_factor > 1:
            h, w = layer_mosaic.shape
            h_t = (h // bin_factor) * bin_factor
            w_t = (w // bin_factor) * bin_factor
            layer_mosaic = (
                layer_mosaic[:h_t, :w_t]
                .reshape(h_t // bin_factor, bin_factor,
                         w_t // bin_factor, bin_factor)
                .mean(axis=(1, 3))
                .astype(np.float32)
            )

        if save_tif:
            if dtp == np.uint8 and 'U8_range' in kwargs:
                U8_min, U8_max = float(kwargs['U8_range'][0]), float(kwargs['U8_range'][1])
                scale = 255.0 / max(U8_max - U8_min, 1e-6)
                np.subtract(layer_mosaic, U8_min, out=layer_mosaic)
                np.multiply(layer_mosaic, scale, out=layer_mosaic)
                np.clip(layer_mosaic, 0, 255, out=layer_mosaic)
                layer_out = layer_mosaic.astype(np.uint8)
            else:
                if dtp == np.float32:
                    layer_out = layer_mosaic          # no copy
                else:
                    layer_out = layer_mosaic.astype(dtp)
            tiff.imwrite(tif_fname, layer_out)
        if return_layer_array:
            return layer_mosaic, layer_id
        else:
            return np.zeros(1), layer_id

    return np.zeros(1), layer_id


def generate_report_mill_rate_montage_xlsx(Mill_Rate_Data_xlsx, **kwargs):
    '''
    Generate Report Plot for mill rate evaluation from XLSX spreadsheet file. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    Mill_Rate_Data_xlsx : str
        Path to the XLSX spreadsheet file containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.
    
    kwargs:
    ----------
    Mill_Volt_Rate_um_per_V : np.float32
        Milling Voltage to Z conversion (µm/V). Default is 31.235258870176065.
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of 2 ints
        Y and X indices of the tile for which the WD fs frame will be evaluated. Default is (0, 0).
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, Mill_Rate_Data_xlsx.replace('.xlsx','_Mill_Rate.png')).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fname
    '''
    saved_kwargs = read_kwargs_xlsx(Mill_Rate_Data_xlsx, 'kwargs Info', **kwargs)
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    tile_id = kwargs.get('tile_id', (0, 0))
    data_dir = saved_kwargs.get("data_dir", '')
    ldm = 70
    data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    if save_png:
        save_fname = kwargs.get('save_fname', os.path.join(data_dir, Mill_Rate_Data_xlsx.replace('.xlsx','_Mill_Rate.png')))
    else:
        save_fname = 'Image not saved'
    if verbose:
        print('Loading kwarg Data')
    Sample_ID = kwargs.get('Sample_ID', saved_kwargs.get('Sample_ID', ''))
    Saved_Mill_Volt_Rate_um_per_V = saved_kwargs.get("Mill_Volt_Rate_um_per_V", 31.235258870176065)
    Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", Saved_Mill_Volt_Rate_um_per_V)
    if verbose:
        print('Loading Working Distance and Milling Y Voltage Data')
    try:
        int_results_all = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name='FIBSEM Data')
    except:
        int_results_all = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name='Milling Rate Data')
        
    int_results = int_results_all.iloc[mosaic_shape[0]*tile_id[0]+tile_id[1]::nxny, :]
    fr = int_results['Frame']/nxny
    WD = int_results['Working Distance (mm)']
    MillingYVoltage = int_results['Milling Y Voltage (V)']

    if verbose:
        print('Generating Plot')
    fs = 12
    fig, axs = plt.subplots(3,1, figsize = (6,10), sharex=True)
    fig.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.05)
    
    for k in np.arange(nxny):
        my_col = plt.get_cmap("gist_rainbow_r")((nxny-k)/(nxny-1))
        WDk = int_results_all.iloc[k::nxny, :]['Working Distance (mm)']
        if k == mosaic_shape[1]*tile_id[0]+tile_id[1]:
            axs[0].plot(fr, WDk, color=my_col, marker='x', markersize=4)
        else:
            axs[0].plot(fr, WDk, color=my_col)
    fr_all = np.repeat(np.array(fr), nxny)
    WD_all = np.array(int_results_all['Working Distance (mm)'])
    WD_all_fit_coef = np.polyfit(fr_all, WD_all, 1)
    WD_fit_all = np.polyval(WD_all_fit_coef, fr)
    axs[0].plot(fr, WD_fit_all, label='All Tiles: Fit, slope = {:.2f} nm/line'.format(WD_all_fit_coef[0]*1.0e6), color='black', linestyle='dashed', linewidth=2)
    axs[0].legend(fontsize=12, loc = 'lower right')
    axs[0].grid(True)
    axs[0].set_ylabel('Working Distance (mm)')
    axs[0].text(0.40, 0.92, 'All Tiles', transform=axs[0].transAxes, fontsize=12)
    axs[0].text(0.2, 1.04, Sample_ID, fontsize = fs, transform=axs[0].transAxes)
    axs[1].plot(fr, WD, label='WD, Exp. Data', color='blue')
    axs[1].grid(True)
    axs[1].set_ylabel('Working Distance (mm)')
    WD_fit_coef = np.polyfit(fr, WD, 1)
    WD_fit = np.polyval(WD_fit_coef, fr)
    axs[1].plot(fr, WD_fit, label='Tile={:d},{:d}: Fit, slope = {:.2f} nm/line'.format(*tile_id, WD_fit_coef[0]*1.0e6), color='red', linestyle='dashed', linewidth=2)
    axs[1].text(0.40, 0.92, 'Tile={:d},{:d}'.format(*tile_id), transform=axs[1].transAxes, fontsize=12)
    axs[1].legend(fontsize=12)

    axs[2].plot(fr, MillingYVoltage, label='Mill. Y Volt. Exp. Data', color='green')
    axs[2].grid(True)
    axs[2].set_ylabel('Milling Y Voltage (V)')
    MV_fit_coef = np.polyfit(fr, MillingYVoltage, 1)
    MV_fit=np.polyval(MV_fit_coef, fr)
    axs[2].plot(fr, MV_fit, label='Fit, slope = {:.3f} nm/line'.format(MV_fit_coef[0]*Mill_Volt_Rate_um_per_V*-1.0e3), color='orange')
    axs[2].legend(fontsize=12)
    axs[2].text(0.02, 0.05, 'Milling Voltage to Z conversion: {:.4f} µm/V'.format(Mill_Volt_Rate_um_per_V), transform=axs[2].transAxes, fontsize=12)
    axs[2].set_xlabel('Frame')
    
    if save_png:
        axs[2].text(-0.12, -0.17, save_fname, fontsize = 5, transform=axs[2].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    display(fig)
    plt.close(fig)
    return save_fname


def generate_report_mill_rate_montage_parquet(Mill_Rate_Data_parquet, **kwargs):
    '''
    Generate Report Plot for mill rate evaluation from Parquet file. (c) G.Shtengel 05/2026 gleb.shtengel@gmail.com

    Parameters:
    ----------
    Mill_Rate_Data_parquet : str
        Path to the Parquet file containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.

    kwargs:
    ----------
    Mill_Volt_Rate_um_per_V : np.float32
        Milling Voltage to Z conversion (um/V). Default is 31.235258870176065.
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of 2 ints
        Y and X indices of the tile for which the WD vs frame will be evaluated. Default is (0, 0).
    data_dir : str
        Directory for saving PNG output. Default is the directory containing Mill_Rate_Data_parquet.
    Sample_ID : str
        Identifier shown above the plot. Default is ''.
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, <parquet stem>_Mill_Rate.png).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fname : str
    '''
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    tile_id = kwargs.get('tile_id', (0, 0))
    data_dir = kwargs.get('data_dir', os.path.dirname(Mill_Rate_Data_parquet))
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else '... ' + data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    Sample_ID = kwargs.get('Sample_ID', '')
    Mill_Volt_Rate_um_per_V = kwargs.get('Mill_Volt_Rate_um_per_V', 31.235258870176065)

    if save_png:
        parquet_stem = os.path.splitext(os.path.basename(Mill_Rate_Data_parquet))[0]
        default_save_fname = os.path.join(data_dir, parquet_stem + '_Mill_Rate.png')
        save_fname = kwargs.get('save_fname', default_save_fname)
    else:
        save_fname = 'Image not saved'

    if verbose:
        print('Loading Working Distance and Milling Y Voltage Data')
    int_results_all = pd.read_parquet(Mill_Rate_Data_parquet)

    int_results = int_results_all.iloc[mosaic_shape[0]*tile_id[0]+tile_id[1]::nxny, :]
    fr = int_results['Frame']/nxny
    WD = int_results['Working Distance (mm)']
    MillingYVoltage = int_results['Milling Y Voltage (V)']

    if verbose:
        print('Generating Plot')
    fs = 12
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    fig.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.05)

    for k in np.arange(nxny):
        my_col = plt.get_cmap("gist_rainbow_r")((nxny-k)/(nxny-1))
        WDk = int_results_all.iloc[k::nxny, :]['Working Distance (mm)']
        if k == mosaic_shape[1]*tile_id[0]+tile_id[1]:
            axs[0].plot(fr, WDk, color=my_col, marker='x', markersize=4)
        else:
            axs[0].plot(fr, WDk, color=my_col)
    fr_all = np.repeat(np.array(fr), nxny)
    WD_all = np.array(int_results_all['Working Distance (mm)'])
    WD_all_fit_coef = np.polyfit(fr_all, WD_all, 1)
    WD_fit_all = np.polyval(WD_all_fit_coef, fr)
    axs[0].plot(fr, WD_fit_all, label='All Tiles: Fit, slope = {:.2f} nm/line'.format(WD_all_fit_coef[0]*1.0e6), color='black', linestyle='dashed', linewidth=2)
    axs[0].legend(fontsize=12, loc='lower right')
    axs[0].grid(True)
    axs[0].set_ylabel('Working Distance (mm)')
    axs[0].text(0.40, 0.92, 'All Tiles', transform=axs[0].transAxes, fontsize=12)
    axs[0].text(0.2, 1.04, Sample_ID, fontsize=fs, transform=axs[0].transAxes)
    axs[1].plot(fr, WD, label='WD, Exp. Data', color='blue')
    axs[1].grid(True)
    axs[1].set_ylabel('Working Distance (mm)')
    WD_fit_coef = np.polyfit(fr, WD, 1)
    WD_fit = np.polyval(WD_fit_coef, fr)
    axs[1].plot(fr, WD_fit, label='Tile={:d},{:d}: Fit, slope = {:.2f} nm/line'.format(*tile_id, WD_fit_coef[0]*1.0e6), color='red', linestyle='dashed', linewidth=2)
    axs[1].text(0.40, 0.92, 'Tile={:d},{:d}'.format(*tile_id), transform=axs[1].transAxes, fontsize=12)
    axs[1].legend(fontsize=12)

    axs[2].plot(fr, MillingYVoltage, label='Mill. Y Volt. Exp. Data', color='green')
    axs[2].grid(True)
    axs[2].set_ylabel('Milling Y Voltage (V)')
    MV_fit_coef = np.polyfit(fr, MillingYVoltage, 1)
    MV_fit = np.polyval(MV_fit_coef, fr)
    axs[2].plot(fr, MV_fit, label='Fit, slope = {:.3f} nm/line'.format(MV_fit_coef[0]*Mill_Volt_Rate_um_per_V*-1.0e3), color='orange')
    axs[2].legend(fontsize=12)
    axs[2].text(0.02, 0.05, 'Milling Voltage to Z conversion: {:.4f} um/V'.format(Mill_Volt_Rate_um_per_V), transform=axs[2].transAxes, fontsize=12)
    axs[2].set_xlabel('Frame')

    if save_png:
        axs[2].text(-0.12, -0.17, save_fname, fontsize=5, transform=axs[2].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    display(fig)
    plt.close(fig)
    return save_fname


def generate_report_SEM_param_mosaic_stack_xlsx(FIBSEM_Data_xlsx, **kwargs):
    '''
    Generate Report Plot SEM parameter vs frame from XLSX spreadsheet file. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    FIBSEM_Data_xlsx : str
        Path to the XLSX spreadsheet file containing the FIBSEM data.
    
    kwargs:
    ----------
    SEM_params : list of str
        SEM parameters to analyze. Options are: 'WD', 'SEMStiX', 'SEMStiY', 'SEMAlnX', 'SEMAlnY'. Default is ['SEMStiX', 'SEMStiY'].
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of 2 ints
        Y and X indices of the tile for which the WD fs frame will be evaluated. Default is (0, 0).
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, Mill_Rate_Data_xlsx.replace('.xlsx','_SEM_params[k].png')).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fnames
    '''
    saved_kwargs = read_kwargs_xlsx(FIBSEM_Data_xlsx, 'kwargs Info', **kwargs)
    SEM_params = kwargs.get('SEM_params', ['SEMStiX', 'SEMStiY'])
    num_SEM_params = len(SEM_params)
    linestyles = kwargs.get('linestyles', ['-', ':', '--', '-.', '-'])
    SEM_keys = []
    for SEM_param in SEM_params:
        if SEM_param == 'WD':
            SEM_keys.append('Working Distance (mm)')
        else:
            SEM_keys.append(SEM_param)
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    tile_id = kwargs.get('tile_id', (0, 0))
    data_dir = saved_kwargs.get("data_dir", '')
    ldm = 70
    data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    if verbose:
        print('Loading kwarg Data')
    Sample_ID = kwargs.get('Sample_ID', saved_kwargs.get('Sample_ID', ''))
    if verbose:
        print('Loading FIBSEM Data')
    try:
        int_results_all = pd.read_excel(FIBSEM_Data_xlsx, sheet_name='FIBSEM Data')
    except:
        int_results_all = pd.read_excel(FIBSEM_Data_xlsx, sheet_name='Milling Rate Data')
        
    int_results = int_results_all.iloc[mosaic_shape[0]*tile_id[0]+tile_id[1]::nxny, :]
    fr = int_results['Frame']/nxny

    if verbose:
        print('Generating Plots')
    save_fnames = []
    for k, SEM_key in enumerate(SEM_keys):
        fs = 12
        fig, axs = plt.subplots(2, 1, figsize = (6, 6), sharex=True)
        fig.subplots_adjust(left=0.12, bottom=0.1, right=0.99, top=0.96, wspace=0.05, hspace=0.05)

        for l in np.arange(nxny):
            my_col = plt.get_cmap("gist_rainbow_r")((nxny-l)/(nxny-1))
            SEMl = int_results_all.iloc[l::nxny, :][SEM_key]
            if l == mosaic_shape[1]*tile_id[0]+tile_id[1]:
                label = SEM_params[k] + ', Tile={:d},{:d}'.format(*tile_id)
                axs[0].plot(fr, SEMl, color=my_col, label=label)
                axs[1].plot(fr, SEMl, color=my_col, label=label)
            else:
                axs[0].plot(fr, SEMl, color=my_col)
        axs[0].legend(fontsize=12, loc = 'lower right')        
        axs[0].text(0.40, 0.92, 'All Tiles', transform=axs[0].transAxes, fontsize=12)
        axs[0].text(0.2, 1.04, Sample_ID, fontsize = fs, transform=axs[0].transAxes)        
        axs[1].text(0.40, 0.92, 'Tile={:d},{:d}'.format(*tile_id), transform=axs[1].transAxes, fontsize=12)
        axs[1].set_xlabel('Frame')
        for ax in axs:
            ax.grid(True)
            ax.set_ylabel(SEM_key)
            ax.legend(fontsize=12, loc = 'lower right')
        if save_png:
            save_fname = kwargs.get('save_fname', os.path.join(data_dir, FIBSEM_Data_xlsx.replace('.xlsx', '_' + SEM_params[k] + '.png')))
            axs[-1].text(-0.12, -0.23, save_fname, fontsize = 5, transform=axs[-1].transAxes)
            fig.savefig(save_fname, dpi=dpi)
        else:
            save_fname = 'Image not saved'
        save_fnames.append(save_fname) 
        display(fig)
        plt.close(fig)
    return save_fnames


def generate_report_SEM_param_mosaic_stack_parquet(FIBSEM_Data_parquet, **kwargs):
    '''
    Generate Report Plot SEM parameter vs frame from Parquet file. (c) G.Shtengel 05/2026 gleb.shtengel@gmail.com

    Parameters:
    ----------
    FIBSEM_Data_parquet : str
        Path to the Parquet file containing the FIBSEM data.

    kwargs:
    ----------
    SEM_params : list of str
        SEM parameters to analyze. Options are: 'WD', 'SEMStiX', 'SEMStiY', 'SEMAlnX', 'SEMAlnY'. Default is ['SEMStiX', 'SEMStiY'].
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of 2 ints
        Y and X indices of the tile for which the WD vs frame will be evaluated. Default is (0, 0).
    data_dir : str
        Directory for saving PNG output. Default is the directory containing FIBSEM_Data_parquet.
    Sample_ID : str
        Identifier shown in the plot title. Default is ''.
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, <parquet stem>_<SEM_param>.png).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fnames : list of str
    '''
    SEM_params = kwargs.get('SEM_params', ['SEMStiX', 'SEMStiY'])
    num_SEM_params = len(SEM_params)
    linestyles = kwargs.get('linestyles', ['-', ':', '--', '-.', '-'])
    SEM_keys = []
    for SEM_param in SEM_params:
        if SEM_param == 'WD':
            SEM_keys.append('Working Distance (mm)')
        else:
            SEM_keys.append(SEM_param)
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    tile_id = kwargs.get('tile_id', (0, 0))
    data_dir = kwargs.get('data_dir', os.path.dirname(FIBSEM_Data_parquet))
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else '... ' + data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    Sample_ID = kwargs.get('Sample_ID', '')
    if verbose:
        print('Loading FIBSEM Data')
    int_results_all = pd.read_parquet(FIBSEM_Data_parquet)
    int_results = int_results_all.iloc[mosaic_shape[0]*tile_id[0]+tile_id[1]::nxny, :]
    fr = int_results['Frame']/nxny

    if verbose:
        print('Generating Plots')
    save_fnames = []
    parquet_stem = os.path.splitext(os.path.basename(FIBSEM_Data_parquet))[0]
    for k, SEM_key in enumerate(SEM_keys):
        fs = 12
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        fig.subplots_adjust(left=0.12, bottom=0.1, right=0.99, top=0.96, wspace=0.05, hspace=0.05)

        for l in np.arange(nxny):
            my_col = plt.get_cmap("gist_rainbow_r")((nxny-l)/(nxny-1))
            SEMl = int_results_all.iloc[l::nxny, :][SEM_key]
            if l == mosaic_shape[1]*tile_id[0]+tile_id[1]:
                label = SEM_params[k] + ', Tile={:d},{:d}'.format(*tile_id)
                axs[0].plot(fr, SEMl, color=my_col, label=label)
                axs[1].plot(fr, SEMl, color=my_col, label=label)
            else:
                axs[0].plot(fr, SEMl, color=my_col)
        axs[0].legend(fontsize=12, loc='lower right')
        axs[0].text(0.40, 0.92, 'All Tiles', transform=axs[0].transAxes, fontsize=12)
        axs[0].text(0.2, 1.04, Sample_ID, fontsize=fs, transform=axs[0].transAxes)
        axs[1].text(0.40, 0.92, 'Tile={:d},{:d}'.format(*tile_id), transform=axs[1].transAxes, fontsize=12)
        axs[1].set_xlabel('Frame')
        for ax in axs:
            ax.grid(True)
            ax.set_ylabel(SEM_key)
            ax.legend(fontsize=12, loc='lower right')
        if save_png:
            default_save_fname = os.path.join(data_dir, parquet_stem + '_' + SEM_params[k] + '.png')
            save_fname = kwargs.get('save_fname', default_save_fname)
            axs[-1].text(-0.12, -0.23, save_fname, fontsize=5, transform=axs[-1].transAxes)
            fig.savefig(save_fname, dpi=dpi)
        else:
            save_fname = 'Image not saved'
        save_fnames.append(save_fname)
        display(fig)
        plt.close(fig)
    return save_fnames


def generate_report_SEM_param_mosaic_layer_xlsx(FIBSEM_Data_xlsx, **kwargs):
    '''
    Generate Report Plot for mill rate evaluation from XLSX spreadsheet file. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    FIBSEM_Data_xlsx : str
        Path to the XLSX spreadsheet file containing the FIBSEM data.
    
    kwargs:
    ----------
    SEM_params : list of str
        SEM parameters to analyze. Options are: 'WD', 'SEMStiX', 'SEMStiY', 'SEMAlnX', 'SEMAlnY'. Default is ['SEMStiX', 'SEMStiY'].
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    frame_id : int
        ID of the frame to show the SEM parameter map over the tile mosaic.
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, Mill_Rate_Data_xlsx.replace('.xlsx','_SEM_param.png')).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fname
    '''
    saved_kwargs = read_kwargs_xlsx(FIBSEM_Data_xlsx, 'kwargs Info', **kwargs)
    SEM_params = kwargs.get('SEM_params', ['SEMStiX', 'SEMStiY'])
    num_SEM_params = len(SEM_params)
    linestyles = kwargs.get('linestyles', ['-', ':', '--', '-.', '-'])
    SEM_keys = []
    Yaxis_title = ''
    fname_repl_suffix = ''
    for SEM_param in SEM_params:
        Yaxis_title = Yaxis_title + SEM_param + ', '
        if SEM_param == 'WD':
            SEM_keys.append('Working Distance (mm)')
            fname_repl_suffix = fname_repl_suffix + '_WD'
        else:
            SEM_keys.append(SEM_param)
            fname_repl_suffix = fname_repl_suffix + '_' + SEM_param
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    frame_id =  kwargs.get('frame_id', -1)
    data_dir = saved_kwargs.get("data_dir", '')
    ldm = 70
    data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    if verbose:
        print('Loading kwarg Data')
    Sample_ID = kwargs.get('Sample_ID', saved_kwargs.get('Sample_ID', ''))
    if verbose:
        print('Loading FIBSEM Data')
    try:
        int_results_all = pd.read_excel(FIBSEM_Data_xlsx, sheet_name='FIBSEM Data')
    except:
        int_results_all = pd.read_excel(FIBSEM_Data_xlsx, sheet_name='Milling Rate Data')

    if verbose:
        print('Generating Plot')
    fs = 12
    fig, axs = plt.subplots(num_SEM_params+1,1, figsize = (6,num_SEM_params*3+1), gridspec_kw={"height_ratios" : [1.5]*num_SEM_params + [2]})
    fig.subplots_adjust(left=0.12, bottom=0.02, right=0.99, top=0.98, wspace=0.05, hspace=0.25)
    
    nz = int(len(int_results_all)/nxny)
    if frame_id==-1:
        frame_id = nz-1
    ny, nx = mosaic_shape
    all_params = []

    for j, SEM_key in enumerate(SEM_keys):
        SEMk = np.array(int_results_all[SEM_key]).reshape(nz, ny, nx)
        all_params.append(SEMk[frame_id])

    all_params = np.array(all_params)
    All_strs = []
    for j in np.arange(ny):
        for i in np.arange(nx):
            loc_str = ''
            for k, SEM_param in enumerate(SEM_params):
                if k==0:
                    loc_str = loc_str + SEM_param + '={:.6f}'.format(all_params[k,j,i])
                else:
                    loc_str = loc_str + '\n' + SEM_param + '={:.6f}'.format(all_params[k,j,i])
            All_strs.append(loc_str)
    All_strs = np.array(All_strs).reshape(mosaic_shape)

    for k, SEM_key in enumerate(SEM_keys):
        for j in np.arange(ny):
            my_col = plt.get_cmap("gist_rainbow_r")((ny-j)/(ny-1))
            label = 'Y Tile = {:d}'.format(j)
            axs[k].plot(all_params[k, j, :], color=my_col, marker='x', markersize=4, label = label)
        axs[k].set_ylabel(SEM_keys[k])
        axs[k].grid(True)
        axs[k].set_xlabel('X Tile #')
        axs[k].legend(fontsize=10, loc = 'lower right')

    axs[-1].axis(False)
    axs[0].set_title(Sample_ID+', frame={:d}'.format(frame_id))
    llw1 = 0.9 / mosaic_shape[1]
    clw = [llw1 for k in np.arange(mosaic_shape[1])]
    tbl = axs[-1].table(cellText = All_strs,
                   colWidths=clw,
                   cellLoc='center',
                   colLoc='center',
                   bbox = [0.02, 0, 0.96, 1.0],
                   zorder=10)
    if save_png:
        save_fname = kwargs.get('save_fname', os.path.join(data_dir, FIBSEM_Data_xlsx.replace('.xlsx',fname_repl_suffix+'_frame{:d}.png'.format(frame_id))))
        axs[-1].text(-0.12, -0.07, save_fname, fontsize = 4, transform=axs[-1].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    else:
        save_fname = 'Image not saved'
    display(fig)
    plt.close(fig)
    return save_fname


def generate_report_SEM_param_mosaic_layer_parquet(FIBSEM_Data_parquet, **kwargs):
    '''
    Generate Report Plot for mill rate evaluation from Parquet file. (c) G.Shtengel 05/2026 gleb.shtengel@gmail.com

    Parameters:
    ----------
    FIBSEM_Data_parquet : str
        Path to the Parquet file containing the FIBSEM data.

    kwargs:
    ----------
    SEM_params : list of str
        SEM parameters to analyze. Options are: 'WD', 'SEMStiX', 'SEMStiY', 'SEMAlnX', 'SEMAlnY'. Default is ['SEMStiX', 'SEMStiY'].
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    frame_id : int
        ID of the frame to show the SEM parameter map over the tile mosaic. Default is -1 (last frame).
    data_dir : str
        Directory for saving PNG output. Default is the directory containing FIBSEM_Data_parquet.
    Sample_ID : str
        Identifier shown in the plot title. Default is ''.
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, <parquet stem>_<SEM_params suffix>_frame<frame_id>.png).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fname : str
    '''
    SEM_params = kwargs.get('SEM_params', ['SEMStiX', 'SEMStiY'])
    num_SEM_params = len(SEM_params)
    linestyles = kwargs.get('linestyles', ['-', ':', '--', '-.', '-'])
    SEM_keys = []
    Yaxis_title = ''
    fname_repl_suffix = ''
    for SEM_param in SEM_params:
        Yaxis_title = Yaxis_title + SEM_param + ', '
        if SEM_param == 'WD':
            SEM_keys.append('Working Distance (mm)')
            fname_repl_suffix = fname_repl_suffix + '_WD'
        else:
            SEM_keys.append(SEM_param)
            fname_repl_suffix = fname_repl_suffix + '_' + SEM_param
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    frame_id = kwargs.get('frame_id', -1)
    data_dir = kwargs.get('data_dir', os.path.dirname(FIBSEM_Data_parquet))
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else '... ' + data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    Sample_ID = kwargs.get('Sample_ID', '')
    if verbose:
        print('Loading FIBSEM Data')
    int_results_all = pd.read_parquet(FIBSEM_Data_parquet)

    if verbose:
        print('Generating Plot')
    fs = 12
    fig, axs = plt.subplots(num_SEM_params+1, 1, figsize=(6, num_SEM_params*3+1),
                            gridspec_kw={"height_ratios": [1.5]*num_SEM_params + [2]})
    fig.subplots_adjust(left=0.12, bottom=0.02, right=0.99, top=0.98, wspace=0.05, hspace=0.25)

    nz = int(len(int_results_all)/nxny)
    if frame_id == -1:
        frame_id = nz-1
    ny, nx = mosaic_shape
    all_params = []

    for j, SEM_key in enumerate(SEM_keys):
        SEMk = np.array(int_results_all[SEM_key]).reshape(nz, ny, nx)
        all_params.append(SEMk[frame_id])

    all_params = np.array(all_params)
    All_strs = []
    for j in np.arange(ny):
        for i in np.arange(nx):
            loc_str = ''
            for k, SEM_param in enumerate(SEM_params):
                if k == 0:
                    loc_str = loc_str + SEM_param + '={:.6f}'.format(all_params[k, j, i])
                else:
                    loc_str = loc_str + '\n' + SEM_param + '={:.6f}'.format(all_params[k, j, i])
            All_strs.append(loc_str)
    All_strs = np.array(All_strs).reshape(mosaic_shape)

    for k, SEM_key in enumerate(SEM_keys):
        for j in np.arange(ny):
            my_col = plt.get_cmap("gist_rainbow_r")((ny-j)/(ny-1))
            label = 'Y Tile = {:d}'.format(j)
            axs[k].plot(all_params[k, j, :], color=my_col, marker='x', markersize=4, label=label)
        axs[k].set_ylabel(SEM_keys[k])
        axs[k].grid(True)
        axs[k].set_xlabel('X Tile #')
        axs[k].legend(fontsize=10, loc='lower right')

    axs[-1].axis(False)
    axs[0].set_title(Sample_ID + ', frame={:d}'.format(frame_id))
    llw1 = 0.9 / mosaic_shape[1]
    clw = [llw1 for k in np.arange(mosaic_shape[1])]
    tbl = axs[-1].table(cellText=All_strs,
                        colWidths=clw,
                        cellLoc='center',
                        colLoc='center',
                        bbox=[0.02, 0, 0.96, 1.0],
                        zorder=10)
    if save_png:
        parquet_stem = os.path.splitext(os.path.basename(FIBSEM_Data_parquet))[0]
        default_save_fname = os.path.join(data_dir,
                                          parquet_stem + fname_repl_suffix + '_frame{:d}.png'.format(frame_id))
        save_fname = kwargs.get('save_fname', default_save_fname)
        axs[-1].text(-0.12, -0.07, save_fname, fontsize=4, transform=axs[-1].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    else:
        save_fname = 'Image not saved'
    display(fig)
    plt.close(fig)
    return save_fname


def generate_report_data_minmax_montage_parquet(minmax_parquet_file, **kwargs):
    '''
    Generate Report Plot for data Min-Max from Parquet file. (c) G.Shtengel 05/2026 gleb.shtengel@gmail.com

    Parameters:
    ----------
    minmax_parquet_file : str
        Path to the Parquet file containing Min-Max data.

    kwargs:
    ----------
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of 2 ints
        Y and X indices of the tile to highlight on the per-tile axis. Default is (0, 0).
    data_dir : str
        Directory for saving PNG output. Default is the directory containing minmax_parquet_file.
    Sample_ID : str
        Identifier shown above the plot. Default is ''.
    thr_min : float
        Min CDF threshold value shown as annotation. Default is 0.0.
    thr_max : float
        Max CDF threshold value shown as annotation. Default is 0.0.
    fit_params : list
        Savitzky-Golay fit parameters [type, window, polyorder] for the sliding bands. Default is ['SG', 101, 3]. Set type to 'None' to skip smoothing.
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, <parquet stem>_Min_Max.png).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fname : str
    '''
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    tile_id = kwargs.get('tile_id', (0, 0))
    data_dir = kwargs.get('data_dir', os.path.dirname(minmax_parquet_file))
    ldm = 70
    data_dir_short = data_dir if len(data_dir) < ldm else '... ' + data_dir[-ldm:]
    verbose = kwargs.get('verbose', False)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    Sample_ID = kwargs.get('Sample_ID', '')
    thr_min = kwargs.get('thr_min', 0.0)
    thr_max = kwargs.get('thr_max', 0.0)
    fit_params = kwargs.get('fit_params', ['SG', 101, 3])

    if save_png:
        parquet_stem = os.path.splitext(os.path.basename(minmax_parquet_file))[0]
        default_save_fname = os.path.join(data_dir, parquet_stem + '_Min_Max.png')
        save_fname = kwargs.get('save_fname', default_save_fname)
    else:
        save_fname = 'Image not saved'

    if verbose:
        print('Loading MinMax Data')
    int_results_all = pd.read_parquet(minmax_parquet_file)

    int_results = int_results_all.iloc[mosaic_shape[0]*tile_id[0]+tile_id[1]::nxny, :]
    frames = int_results['Frame']/nxny
    frame_min = np.array(int_results['Min'])
    frame_max = np.array(int_results['Max'])
    data_min_glob = np.min(frame_min)
    data_max_glob = np.max(frame_max)

    if verbose:
        print('Generating Plots')
    fs = 12

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    fig.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.05)

    for k in np.arange(nxny):
        my_col = plt.get_cmap("gist_rainbow_r")((nxny-k)/(nxny-1))
        framek_min = int_results_all.iloc[k::nxny, :]['Min']
        framek_max = int_results_all.iloc[k::nxny, :]['Max']
        if k == mosaic_shape[1]*tile_id[0]+tile_id[1]:
            axs[0].plot(frames, framek_min, color=my_col, marker='x', markersize=4)
            axs[1].plot(frames, framek_max, color=my_col, marker='x', markersize=4)
        else:
            axs[0].plot(frames, framek_min, color=my_col, linewidth=0.5)
            axs[1].plot(frames, framek_max, color=my_col, linewidth=0.5)
    axs[0].set_ylabel('All Tiles Minima Values')
    axs[1].set_ylabel('All Tiles Maxima Values')

    if fit_params[0] != 'None':
        sv_apert = min([fit_params[1], len(frames)//8*2+1])
        if verbose:
            print('Using fit_params: ', 'SG', sv_apert, fit_params[2])
        sliding_min = savgol_filter(frame_min.astype(np.double), sv_apert, fit_params[2])
        sliding_max = savgol_filter(frame_max.astype(np.double), sv_apert, fit_params[2])
    else:
        if verbose:
            print('Not smoothing the Min/Max data')
        sliding_min = frame_min.astype(np.double)
        sliding_max = frame_max.astype(np.double)

    axs[0].text(0.2, 1.04, Sample_ID, fontsize=fs, transform=axs[0].transAxes)
    axs[2].plot(frame_min, 'b', linewidth=1, label='Frame Minima')
    axs[2].plot(sliding_min, 'b', linewidth=2, linestyle='dotted', label='Sliding Minima')
    axs[2].plot(frame_max, 'r', linewidth=1, label='Frame Maxima')
    axs[2].plot(sliding_max, 'r', linewidth=2, linestyle='dotted', label='Sliding Maxima')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Tile ({:d},{:d}) Minima and Maxima Values'.format(*tile_id))
    dxn = (data_max_glob - data_min_glob)*0.1
    axs[2].set_ylim((data_min_glob - dxn, data_max_glob+dxn))
    xminmax = [0, len(frame_min)]
    y_min = [data_min_glob, data_min_glob]
    y_max = [data_max_glob, data_max_glob]
    axs[2].plot(xminmax, y_min, 'b', linestyle='--')
    axs[2].plot(xminmax, y_max, 'r', linestyle='--')
    axs[2].text(len(frame_min)/20.0, data_min_glob-dxn/1.75, 'data_min_glob={:.1f}'.format(data_min_glob), fontsize=fs-2, c='b')
    axs[2].text(len(frame_min)/20.0, data_max_glob+dxn/2.25, 'data_max_glob={:.1f}'.format(data_max_glob), fontsize=fs-2, c='r')
    axs[2].text(len(frame_min)/20.0, data_min_glob+dxn*4.5, 'thr_min={:.1e}'.format(thr_min), fontsize=fs-2, c='b')
    axs[2].text(len(frame_min)/20.0, data_min_glob+dxn*5.5, 'thr_max={:.1e}'.format(thr_max), fontsize=fs-2, c='r')
    for ax in axs:
        ax.grid(True)
    if save_png:
        axs[2].text(-0.12, -0.17, save_fname, fontsize=5, transform=axs[2].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    display(fig)
    plt.close(fig)
    return save_fname


def reformat_outliers_data(outliers):
    res = []
    for outlier in outliers:
        for frame in outlier[1]:
            res.append([frame, outlier[0]])
    if len(res) == 0:
        return np.empty((0, 2), dtype=int)
    res = np.array(res)
    return res[res[:, 0].argsort()]


def generate_outliers_report(outliers, fls, **kwargs):
    '''
    Generate quick summary view of potential outliers.
    Parameters:
    -----------
    outliers : list of [frame_id, tile_id]
        List of potential outliers
    fls : 2D array of filenames

    kwargs:
    -----------
    save_outlier_thumbnails : bool
        If True (default), outlier RawImageA thumbnails will be saved
    data_dir : path
        data directory
    outliers_thumbnails_folder : str
        sub-directory name (will be created inside data_dir). Default is 'outliers_thumbnails'
    bin_factor : int
        Binning factor. Default = 10
    dpi : int
        DPI for PNG. Default is 150.
    verbose : boolean
        Display intermediate results. Default is False.
    '''
    save_outlier_thumbnails = kwargs.get('save_outlier_thumbnails', True)
    data_dir = kwargs.get("data_dir", '')
    outliers_thumbnails_folder = kwargs.get('outliers_thumbnails_folder', 'outliers_thumbnails')
    if save_outlier_thumbnails:
        save_folder = os.path.join(data_dir, outliers_thumbnails_folder)
        os.makedirs(save_folder, exist_ok=True)
    bin_factor = kwargs.get('bin_factor', 10)
    verbose = kwargs.get('verbose', True)
    dpi = kwargs.get('dpi', 150)

    for outlier in tqdm(outliers, desc='Building Outlier Report'):
        outlier_fnm = fls[*outlier]
        img = FIBSEM_frame(outlier_fnm).RawImageA
        sy, sx = img.shape
        img_reshaped = img[0:sy//bin_factor*bin_factor, 0:sx//bin_factor*bin_factor].reshape(sy//bin_factor, bin_factor, sx//bin_factor, bin_factor)
        img_binned = np.mean(np.mean(img_reshaped, axis=3), axis=1)
        vmin, vmax = get_min_max_thresholds(img_binned, disp_res=False)
        fx = 5
        fy = fx / sx * sy
        if verbose:
            print('Outlier Frame: {:d} Tile: {:d}  '.format(*outlier),outlier_fnm)
        fig, ax = plt.subplots(1,1, figsize=(fx, fy))
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.96, wspace=0.05, hspace=0.05)
        ax.imshow(img_binned, vmin = vmin, vmax=vmax, cmap='Greys')
        ax.set_title('Frame: {:d} Tile: {:d}  '.format(*outlier) + outlier_fnm, fontsize=6)
        ax.axis(False)
        if save_outlier_thumbnails:
            thumbnail_fnm_short = os.path.splitext(os.path.split(outlier_fnm)[-1])[0]+'_thumbnail.png'
            thumbnail_fnm = os.path.join(save_folder, thumbnail_fnm_short)
            fig.savefig(thumbnail_fnm, dpi=dpi)
        display(fig)
        plt.close(fig)

    
def analyze_minmax_outliers_montage_parquet(minmax_parquet_file, **kwargs):
    '''
    Generate Report Plot for data Min-Max with outlier marking from Parquet file. (c) G.Shtengel 05/2026 gleb.shtengel@gmail.com

    Parameters:
    -----------
    minmax_parquet_file : str
        Path to the Parquet file containing Min-Max data.

    kwargs:
    -----------
    sigma_thr : float
        Threshold (multiplied by sigma) for outlier determination. Default is 6.0 (6-sigma outliers).
    mosaic_shape : tuple or list of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of 2 ints
        Y and X indices of the tile for which the WD fs frame will be evaluated. Default is (0, 0).
    data_dir : str
        Directory for saving PNG output. Default is the directory containing minmax_parquet_file.
    Sample_ID : str
        Identifier shown above the plot. Default is ''.
    fit_params : list
        Savitzky-Golay fit parameters [type, window, polyorder]. Default is ['SG', 11, 3]. Set type to 'None' to use mean-based outlier detection.
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, <parquet stem>_Min_Max_Outliers.png).
    verbose : boolean
        Display intermediate results. Default is False.
    mark_outliers : boolean
        If True (default), each outlier is marked with "x" and its frame and tile number are printed next to "x".

    Returns:
    ----------
    outliers_min, outliers_max
    '''
    sigma_thr = kwargs.get('sigma_thr', 6.0)
    mosaic_shape = kwargs.get('mosaic_shape', (1, 1))
    nxny = np.prod(mosaic_shape)
    data_dir = kwargs.get('data_dir', os.path.dirname(minmax_parquet_file))
    verbose = kwargs.get('verbose', False)
    mark_outliers = kwargs.get('mark_outliers', True)
    save_png = kwargs.get('save_png', True)
    dpi = kwargs.get('dpi', 300)
    Sample_ID = kwargs.get('Sample_ID', '')
    fit_params = kwargs.get('fit_params', ['SG', 11, 3])

    if save_png:
        parquet_stem = os.path.splitext(os.path.basename(minmax_parquet_file))[0]
        default_save_fname = os.path.join(data_dir, parquet_stem + '_Min_Max_Outliers.png')
        save_fname = kwargs.get('save_fname', default_save_fname)
    else:
        save_fname = 'Image not saved'

    if verbose:
        print('Loading MinMax Data')
    int_results_all = pd.read_parquet(minmax_parquet_file)

    frames = np.array(int_results_all.iloc[0::nxny, :]['Frame'])//nxny

    if verbose:
        print('Generating Plots')
    fs = 12
    fsmark = 6

    fig, axs = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    fig.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.05)

    if fit_params[0] != 'None':
        sv_apert = min([fit_params[1], len(frames)//8*2+1])
        if verbose:
            print('Using fit_params: ', 'SG', sv_apert, fit_params[2])

    outliers_min = []
    outliers_max = []
    for k in np.arange(nxny):
        my_col = plt.get_cmap("gist_rainbow_r")(0.0 if nxny == 1 else (nxny-k)/(nxny-1))
        framek_min = np.array(int_results_all.iloc[k::nxny, :]['Min'])
        framek_max = np.array(int_results_all.iloc[k::nxny, :]['Max'])
        axs[0].plot(frames, framek_min, color=my_col, linewidth=0.5)
        axs[1].plot(frames, framek_max, color=my_col, linewidth=0.5)
        if fit_params[0] != 'None':
            sliding_min = savgol_filter(framek_min.astype(np.double), sv_apert, fit_params[2])
            sliding_max = savgol_filter(framek_max.astype(np.double), sv_apert, fit_params[2])
        else:
            sliding_min = np.full_like(framek_min, np.mean(framek_min), dtype=np.double)
            sliding_max = np.full_like(framek_max, np.mean(framek_max), dtype=np.double)
        framek_min_delta = framek_min - sliding_min
        framek_min_std = np.std(framek_min_delta)
        outliersk_min = np.where(np.abs(framek_min_delta) > framek_min_std * sigma_thr)[0]
        if len(outliersk_min) > 0:
            outliers_min.append([k, outliersk_min])
        framek_max_delta = framek_max - sliding_max
        framek_max_std = np.std(framek_max_delta)
        outliersk_max = np.where(np.abs(framek_max_delta) > framek_max_std * sigma_thr)[0]
        if mark_outliers:
            axs[0].plot(frames[outliersk_min], framek_min[outliersk_min], color=my_col, marker='x', markersize=4, linestyle='')
            for outlier_k_min in outliersk_min:
                axs[0].text(frames[outlier_k_min], framek_min[outlier_k_min], '{:d}, {:d}'.format(k, frames[outlier_k_min]), fontsize=fsmark)
            axs[1].plot(frames[outliersk_max], framek_max[outliersk_max], color=my_col, marker='x', markersize=4, linestyle='')
            for outlier_k_max in outliersk_max:
                axs[1].text(frames[outlier_k_max], framek_max[outlier_k_max], '{:d}, {:d}'.format(k, frames[outlier_k_max]), fontsize=fsmark)
        if len(outliersk_max) > 0:
            outliers_max.append([k, outliersk_max])
    outliers_min = reformat_outliers_data(outliers_min)        # bug fix: xlsx version passed (outliers, mosaic_shape) but reformat_outliers_data takes 1 arg, causing TypeError when outliers were found
    outliers_max = reformat_outliers_data(outliers_max)        # bug fix: same as above

    axs[0].set_ylabel('All Tiles Minima Values')
    axs[1].set_ylabel('All Tiles Maxima Values')
    axs[1].set_xlabel('Frame')

    axs[0].text(0.2, 1.04, Sample_ID, fontsize=fs, transform=axs[0].transAxes)
    for ax in axs:
        ax.grid(True)
    if save_png:
        axs[1].text(-0.12, -0.17, save_fname, fontsize=5, transform=axs[1].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    display(fig)
    plt.close(fig)
    return outliers_min, outliers_max


def compute_tile_overlap_intensities(params):
    '''
    DASK worker: load one tile, compute mean/percentile over a list of sub-ROIs.
    ©G.Shtengel gleb.shtengel@gmail.com

    params : [fname, rois, kwargs]
        fname  : str         path to tile file (.dat or .tif)
        rois   : list of (roi_id, x_min, x_max, y_min, y_max)
                 roi_id is opaque to this worker (the caller keys on it).
        kwargs : dict with keys 'ftype', 'method' ('mean' or 'percentile'),
                 'percentile' (int, used when method == 'percentile').

    Returns
    -------
    dict {roi_id: float}
    '''
    fname, rois, kwargs = params
    ftype  = kwargs['ftype']
    method = kwargs['method']
    p      = kwargs.get('percentile', 50)
    img = FIBSEM_frame(fname, ftype=ftype, calculate_scaled_images=False).RawImageA
    out = {}
    for roi_id, xi, xa, yi, ya in rois:
        sub = img[yi:ya, xi:xa]
        if sub.size == 0:
            out[roi_id] = np.nan
        elif method == 'mean':
            out[roi_id] = float(np.mean(sub))
        else:
            out[roi_id] = float(np.percentile(sub, p))
    return out


class FIBSEM_mosaic_dataset: 
    '''
    A class representing a stack of FIB-SEM mosaics (montages) - multiple z-panes consisting of multiple tiles.
    ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    Contains the info/settings on the FIB-SEM montage and the procedures that can be performed on it.

    Attributes:
    ----------
    fls : array of str
        filenames for the individual data frames in the set
    data_dir : str
        data directory (path)
    grid : str
        grid for default tiles positions. Default is 'rect' - rectilinear grid, typical for FIB-SEM. Another options is 'hex' - hexagonal, typical for MSEM
    index_pairs : array of pairs of absolute (in 1D sense of fls.ravel()) tile indices. Auto-determined during initialization, depends on grid setting.
        if grid == 'rect':  index_pairs = np.array(col_ind).reshape((row, 2))
    Sample_ID : str
            Sample ID
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    PixelSize : np.float32
        pixel size in nm. This is inherited from FIBSEM_frame object. Default is 8.0
    voxel_size : rec.array(( np.float32,  np.float32,  np.float32), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        voxel size in nm. Default is isotropic (PixelSize, PixelSize, PixelSize)
    Scaling : 2D array of floats
        scaling parameters allowing to convert I16 data into actual electron counts 
    fnm_reg : str
        filename for the final registered dataset
    use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    thr_min : np.float32
        CDF threshold for determining the minimum data value
    thr_max : np.float32
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    sliding_minmax : boolean
        if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
        if False - same data_min_glob and data_max_glob will be used for all files
    TransformType : object reference
        Transformation model used by SIFT for determining the transformation matrix from Key-Point pairs.
        Choose from the following options:
            ShiftTransform - only x-shift and y-shift
            RotationShiftTransform - x-shift, y-shift, rotation
            XScaleShiftTransform  -  x-scale, x-shift, y-shift
            ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
            AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
            RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
    l2_matrix : 2D np.float32 array
        matrix of regularization (shrinkage) parameters
    targ_vector : 1D np.float32 array
        target vector for regularization
    solver : str
        Solver used for SIFT ('RANSAC' or 'LinReg')
    RANSAC_initial_fraction : float
        Fraction of data points for initial RANSAC iteration step. Default is 0.005.
    drmax : float
        In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
        In the case of 'LinReg' - outlier threshold for iterative regression
    max_iter : int
        Max number of iterations in the iterative procedure above (RANSAC or LinReg)
    BFMatcher : boolean
        If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches : boolean
        If True, matches will be saved into individual files
    SIFT_nfeatures : int
        SIFT library default is 0. The number of best features to retain.
        The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
    SIFT_nOctaveLayers : int
        SIFT library default  is 3. The number of layers in each octave.
        3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
    SIFT_contrastThreshold : double
        SIFT library default  is 0.04. The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        The larger the threshold, the less features are produced by the detector.
        The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
        When nOctaveLayers is set to default and if you want to use the value used in
        D. Lowe paper (0.03), set this argument to 0.09.
    SIFT_edgeThreshold : double
        SIFT library default  is 10. The threshold used to filter out edge-like features.
        Note that its meaning is different from the contrastThreshold,
        i.e. the larger the edgeThreshold, the less features are filtered out
        (more features are retained).
    SIFT_sigma : double
        SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
        If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    dtp : Data Type
        Python data type for saving. Default is np.int16, the other option currently is np.uint8.
    zbin_factor : int
        binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
    flipY : boolean
        If True, the data will be flipped along Y-axis. Default is False.
    preserve_scales : boolean
        If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params below.
    fit_params : list
        Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
        Other options are:
            ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
            ['PF', 2]  - use polynomial fit (in this case of order 2)
    interpolation : int
        The order of interpolation as defined in cv2. The options are:
                                                                #    cv2.INTER_AREA    Uses pixel area relation for resampling, which effectively minimizes distortion and avoids aliasing artifacts, yielding high-quality results for reduced image sizes.
                                                                #    cv2.INTER_CUBIC    Uses bicubic interpolation (based on 4x4 neighboring pixels) to produce smooth, high-quality results. It is slower than INTER_LINEAR.
                                                                #    cv2.INTER_LINEAR   Uses bilinear interpolation (based on 2x2 neighboring pixels), offering a good balance of speed and visual quality for most general resizing tasks.
                                                                #    cv2.INTER_NEAREST  Uses the nearest neighbor, picking the value of the closest pixel. It is quick but results in a blocky, pixelated output, and is generally used for specific cases like segmentation masks.
                                                                #    cv2.INTER_LANCZOS4  Uses a Lanczos kernel with an 8x8 neighborhood, providing the best quality, but it is the slowest method.
    subtract_linear_fit : [boolean, boolean]
        List of two Boolean values for two directions: X- and Y-.
        If True, the linear slopes along X- and Y- directions (respectively)
        will be subtracted from the cumulative shifts.
        This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
    pad_edges : boolean
        If True, the data will be padded before transformation to avoid clipping.
    perform_deformation : boolean
        If True - the data is deformed (in addition to transformation defined above) using the deformation field data defined below
    deformation_type : str
        Options are:
            'post_1DY'  - Default. Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
            'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
    deformation_sigma :  list of 1 or two floats.
        Gaussian width of smoothing (units of pixels). Default is 50.
    ImgB_fraction : float
            fractional ratio of Image B to be used for constructing the fused image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
    evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.

    Methods:
    ----------
    find_tile_pairs(layer_id, tile_id)
        Searches self.index_pairs for all pairs containing the tile. Reports transformation records for each found pair. 

    save_parameters(**kwargs)
        Save transformation attributes and parameters (including transformation matrices)

    evaluate_FIBSEM_statistics(**kwargs)
        Evaluates parameters of FIBSEM data set (data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).

    extract_keypoints(**kwargs):
        Extract Key-Points and Descriptors

    analyze_kpt_statistics(**kwargs):
        Analyze key-point statistic and report suspect outliers.

    determine_transformations_SIFT(self, **kwargs)
        Determine transformation matrices for frame pairs using SIFT. 

    SIFT_evaluation(index_pair, **kwargs)
        Evaluate SIFT performance on a given index_pair.

    determine_transformations_ECC(**kwargs)
        Determine transformation matrices for frame pairs using ECC. Uses find_Transform_ECC(img1, img2, **kwargs).

    solve_stack_stitching(**kwargs)
        Solve mosaic stack stitching (perform bundle optimization).

    solve_intensity_normalization(**kwargs)
        Solve mosaic stack intensity matching (perform bundle optimization).

    generate_transformation_report(**kwargs)
        Generate Report Plot for transformation summary.

    assemble_layer_mosaic(layer_id, **kwargs)
        Assemble layer mosaic based on transformation matrices for each tile. Options to save snapshot, save mosaic as FIBSEM_frame (dat file) or save images as JPG or PNG.

    save_stack(**kwargs)
        Assemble all layers based on transformation matrices for each tile and save them into stack.

    '''
    
    def __init__(self, fls, **kwargs):
        '''
        Initializes (or recalls) an instance of  FIBSEM_mosaic_dataset object. ©G.Shtengel 12/2025 gleb.shtengel@gmail.com

        Parameters:
        ----------
        fls : 2D or 3D array of str
            Filenames for the individual data frames in the stack of montages.

        kwargs:
        ---------
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif).
        data_dir : str
            Data directory (path).
        image_coordinates_files : List of str
            Paths to a whitespace-delimited text files specifying tile coordinates,
            one tile per line: filename  X  Y  (additional columns ignored).
            X and Y are the stage coordinates of the tile's first pixel in pixel units;
            Tiles are matched to fls[0].ravel() by basename.
            If empty string '' (default), tile positions are read from the .dat file        headers (Option 1).
        metadata_file : str
            Path to a text file with MSEM acquisition metadata. Will be parsed using parse_metadata_file(filename).
        grid : str
            grid for default tiles positions. Default is 'rect' - rectilinear grid, typical for FIB-SEM. Another options is 'hex' - hexagonal, typical for MSEM
        fnm_mosaic_stack : str
            Filename for registered mosaic stack. Default is os.path.splitext(os.path.split(self.fls.ravel()[0])[1])[0][0:-5] + 'mosaic_stack.mrc'
        recall_parameters : boolean
            If True and dump_filename kwarg points to a valid binary file, will recall the dataset saved into that dump_filename. Default is False.
        dump_filename : str
            Filename (full path) to a binary dump file with saved dataset attributes. If dump_filename points to a valid binary file the data set saved in that file will be recalled. Default is empty string ''.
        memory_profiling : boolean
            Perform memory profiling during the data load and output it. Default is False.
        max_futures : int
            max number of running DASK futures. Default is 50000.
        intralayer_weight : np.float32, default 1.0
            Weight for pairwise constraints within a single Z-layer.
        interlayer_weight : np.float32, default 100.0
            Weight for pairwise constraints for tiles between adjacent Z-layers.(100–10000 typical).
        add_reverse_edges : bool, default False
            If True, adds both (i->j) and (j->i) with same weight (increases robustness).
        diagonal_exclusion_threshold : np.float32
            A criteria for exclusion of "diagonal" neighbours if the sum of margins in X- and Y- directions is larger than a sum of X- and Y- tiles times this number, the pair is NOT considered.
        shape : tuple of two int (self.ny_tiles, self.nx_tiles)
            The program will try to auto-determine the shape, but it can be set explicitly.
                # self.ny_tiles  - # of rows per layer (# of tiles along Y-axis)
                # self.nx_tiles  - # of columns per layer(# of tiles along X-axis)
        EightBit : int
            If 1 then the data is assumed uint8, otherwise int16
        U8_conversion : str
            Range selection for U8 conversion. Options are: 'global', 'sliding', and 'local'. Default is 'local'.
        left_crop : int
            left image margin to be cropped off BEFORE distortion correction (via deformation field) is applied. Default is 0.
        deformation_field : 3D array
            Array with dimensions (YResolution, XResolution - left_crop, 2). Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
            Deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3.
        Sample_ID : str
            Sample ID.
        PixelSize : np.float32
            Pixel size in nm. Default is determined from the frame metadata. If that is not available, default is 8.0.
        Scaling : 2D array of floats
            Scaling parameters allowing to convert I16 data into actual electron counts.
        thr_min : np.float32
            CDF threshold for determining the minimum data value. Default is 1e-3.
        thr_max : np.float32
            CDF threshold for determining the maximum data value. Default is 1e-3.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is 256.
        sliding_minmax : boolean
            If True - data min and max will be taken from data_min_sliding and data_max_sliding arrays. Default is True.
            If False - same data_min_glob and data_max_glob will be used for all files.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Default is ['SG', len(fls)//100+1, 3].
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        SIFT_nfeatures : int
            The number of best features to retain. SIFT library default is 0 (all features retained).
            The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        SIFT_nOctaveLayers : int
            The number of layers in each octave. SIFT library default is 3.
            3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        SIFT_contrastThreshold : double
            The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. SIFT library default is 0.025.
            The larger the threshold, the less features are produced by the detector.
            The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
            When nOctaveLayers is set to default and if you want to use the value used in
            D. Lowe paper (0.03), set this argument to 0.09.
        SIFT_edgeThreshold : double
            The threshold used to filter out edge-like features. SIFT library default is 10.
            Note that its meaning is different from the contrastThreshold,
            i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        SIFT_sigma : double
            The sigma of the Gaussian applied to the input image at the octave #0. SIFT library default is 1.6.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        BFMatcher : boolean
            If True, the BF Matcher is used for Key-Point matching, otherwise FLANN will be used. Default is False.
        save_matches : boolean
            If True, matches will be saved into individual files. Default is True.
        TransformType : object reference
            Transformation model used for determining the transformation matrix from Key-Point pairs.
            Choose from the following options (default is ShiftTransform):
                ShiftTransform - only x-shift and y-shift
                RotationShiftTransform - x-shift, y-shift, rotation
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D np.float32 array
            Matrix of regularization (shrinkage) parameters (applicable only if RegularizedAffineTransform is used). Default is 1e-5.
        targ_vector : 1D np.float32 array
            Target vector for regularization (applicable only if RegularizedAffineTransform is used). Default is [1, 0, 0, 0, 1, 0] for a target transformation that is shift only: Sxx=Syy=1, Sxy=Syx=0.
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg'). Default is 'RANSAC'.
        RANSAC_initial_fraction : float
            Fraction of data points for initial RANSAC iteration step. Default is 0.005.
        Lowe_Ratio_Threshold : float
            Threshold for Lowe's Ratio Test. Default is 0.7.
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression.
            Default is 1.5.
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg). Default is 1000.
        SIFT_nmatches_min : int
            Min number of matches for the transformation to be considered valid. Default is 5.
        interpolation : int
            The order of interpolation as defined in cv2. The options are:
                                                                    #    cv2.INTER_AREA    Uses pixel area relation for resampling, which effectively minimizes distortion and avoids aliasing artifacts, yielding high-quality results for reduced image sizes.
                                                                    #    cv2.INTER_CUBIC    Uses bicubic interpolation (based on 4x4 neighboring pixels) to produce smooth, high-quality results. It is slower than INTER_LINEAR.
                                                                    #    cv2.INTER_LINEAR   Uses bilinear interpolation (based on 2x2 neighboring pixels), offering a good balance of speed and visual quality for most general resizing tasks.
                                                                    #    cv2.INTER_NEAREST  Uses the nearest neighbor, picking the value of the closest pixel. It is quick but results in a blocky, pixelated output, and is generally used for specific cases like segmentation masks.
                                                                    #    cv2.INTER_LANCZOS4  Uses a Lanczos kernel with an 8x8 neighborhood, providing the best quality, but it is the slowest method.
        dtp : Data Type
            Python data type for saving. Default is np.int16.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping. 
        disp_res : boolean
            If False, the intermediate printouts will be suppressed. Default is True.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is True.
        '''
        memory_profiling = kwargs.get('memory_profiling', False)
        verbose = kwargs.get('verbose', True)
        if memory_profiling:
            rss_before, vms_before, shared_before = get_process_memory()
            start_time = time.time()

        self.fls = np.array(fls)
        self.max_futures = kwargs.get('max_futures', 50000)
        # ---- Early recall path -----------------------------------------------
        # If a parameter dump exists and recall_parameters=True, restore the
        # full saved state and skip the expensive init (reading image coords,
        # determining intra/inter-layer pairs, building image pair list,
        # determining pair overlap bounds, etc.).
        if kwargs.get("recall_parameters", False):
            dump_filename = kwargs.get("dump_filename", '')
            try:
                with open(dump_filename, 'rb') as f:
                    dump_data = pickle.load(f)
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Loading saved parameters from dump: ' + dump_filename)
                for key in tqdm(dump_data, desc='Recalling the data set parameters',
                                display=verbose):
                    setattr(self, key, dump_data[key])
                # max_futures is a runtime / per-session knob — let the kwarg override the dump.
                self.max_futures = kwargs.get('max_futures', getattr(self, 'max_futures', 50000))
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Recalled FIBSEM_mosaic_dataset instance from dump (skipped full init).')
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Initialized FIBSEM_mosaic_dataset instance:')
                    print('Total number of tile files: {:d}'.format(len(self.fls.ravel())))
                    print('Number of tiles per Z-layer: {:d}'.format(self.n_tiles_per_layer))
                    print('Number of Z-slices (nz_tiles): {:d}'.format(self.nz_tiles))
                    print('')
                    print('Total number of left-right intra-layer pairs: ', self.nh)
                    print('Total number of up-down intra-layer pairs: ', self.nv)
                    print('Total number of intra-layer pairs: ', self.nh + self.nv)
                    print('Total number of inter-layer pairs: ', self.nl)
                    print('Total number of pairwise transformations : {:d}'.format(self.C))
                return                                # Skip all expensive setup below.

            except Exception as ex:
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Failed to load dump ({}); falling back to full init: '.format(dump_filename) + str(ex))

        # Standard initialization path.
        fname0 = self.fls.ravel()[0]
        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Started data set initialization')

        # Try to auto-detect image coordinates files
        image_coordinates_files_default = [os.path.join(os.path.split(os.path.split(fl.ravel()[0])[0])[0], 'full_image_coordinates.txt') for fl in self.fls]
        if len(image_coordinates_files_default)>0 and os.path.exists(image_coordinates_files_default[0]):
            image_coordinates_files = kwargs.get('image_coordinates_files', image_coordinates_files_default)
            if verbose:
                print('Will use image coordinates files: ', image_coordinates_files[0], ', ... , ', image_coordinates_files[-1])
        else:
            image_coordinates_files = kwargs.get('image_coordinates_files', '')
        self.image_coordinates_files = image_coordinates_files

        # Try to auto-detect metadata file
        metadata_file_default = os.path.join(os.path.split(fname0)[0], 'metadata.txt')
        if os.path.exists(metadata_file_default):
            metadata_file = kwargs.get('metadata_file', metadata_file_default)
            if verbose:
                print('Will use metadata file: ', metadata_file)
        else:
            metadata_file = kwargs.get('metadata_file', '')
        self.metadata_file = metadata_file
        self.data_dir = kwargs.get('data_dir', os.path.split(fname0)[0])
        def_ftype = 0
        fname_suff = Path(fname0).suffix.lower()
        if fname_suff == '.tif' or fname_suff == '.tiff':
            def_ftype = 1
        if fname_suff == '.png':
            def_ftype = 2
        #print('filename suffix: ',fname_suff, ', default filetype: ', def_ftype)
        self.ftype = kwargs.get("ftype", def_ftype) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files, ftype=2 for PNG files
        self.intralayer_weight = kwargs.get('intralayer_weight', 1.0)
        self.interlayer_weight = kwargs.get('interlayer_weight', 100.0)
        self.add_reverse_edges = kwargs.get('add_reverse_edges', False)
        self.diagonal_exclusion_threshold = kwargs.get('diagonal_exclusion_threshold', 0.75)
        self.U8_conversion = kwargs.get('U8_conversion', 'local')
        self.left_crop = kwargs.get('left_crop', 0)
        self.deformation_field = kwargs.get('deformation_field', np.nan)
        if self.ftype == 0:
            test_frame = FIBSEM_frame(fname0, ftype = self.ftype, calculate_scaled_images=False, read_header_only=True)
            self.MachineID = test_frame.MachineID
            self.FileVersion = test_frame.FileVersion
            self.ScanRate = test_frame.ScanRate
            self.Oversampling = test_frame.Oversampling
            self.WD = test_frame.WD
            self.FIBFocus = test_frame.FIBFocus
            self.FIBProb = test_frame.FIBProb
            self.EHT = test_frame.EHT
            self.SEMCurr = test_frame.SEMCurr
            self.XResolution = kwargs.get("XResolution", test_frame.XResolution)
            self.YResolution = kwargs.get("YResolution", test_frame.YResolution)
            self.XResolutions = kwargs.get('XResolutions', np.full(len(fls[0]), test_frame.XResolution))
            self.YResolutions = kwargs.get('YResolutions', np.full(len(fls[0]), test_frame.YResolution))
            if hasattr(test_frame, 'PixelSize'):
                self.PixelSize = kwargs.get("PixelSize", test_frame.PixelSize)
            else:
                self.PixelSize = kwargs.get("PixelSize", 8.0)
            self.DetA = test_frame.DetA
            self.DetB = test_frame.DetB
            self.Notes = test_frame.Notes
            self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
            if self.DetB == 'None':
                self.ImgB_fraction = 0.0
            self.BrightnessA = test_frame.BrightnessA 
            self.BrightnessB = test_frame.BrightnessB
            self.ContrastA = test_frame.ContrastA
            self.ContrastB = test_frame.ContrastB
            self.Sample_ID = kwargs.get("Sample_ID", test_frame.Sample_ID)
            self.EightBit = kwargs.get("EightBit", test_frame.EightBit)
            # try to auto-determine shape and adjacent pairs
            # self.nz_tiles  - # of layers (# of tiles along Z-axis)
            # self.ny_tiles  - # of rows per layer (# of tiles along Y-axis)
            # self.nx_tiles  - # of columns per layer(# of tiles along X-axis)
            try:
                tile_string = os.path.splitext(os.path.split(self.fls.ravel()[-1])[1])[0][-5:].split('-')    
                auto_ny_tiles = int(tile_string[1])+1
                auto_nx_tiles = int(tile_string[2])+1
                auto_shape = (auto_ny_tiles, auto_nx_tiles)
            except:
                if verbose:
                    print('Could not auto-determine the shape, and therefore the montage size and the adjacent tile pairs')
                    print('Define the montage size (self.Xsize, self.Ysize) manually')
                    print('Define the adjacent tile pairs (self.adjacent_pairs - list of indices of files of the adjacent tiles) manually')
                auto_shape = (1, 1)
            self.shape = kwargs.get('shape', auto_shape)
            self.ny_tiles, self.nx_tiles = self.shape
        
        if self.ftype == 2 and metadata_file:
            metadata = parse_metadata_file(metadata_file)
            test_frame = FIBSEM_frame(fname0, ftype = 2)
            self.FileVersion = test_frame.FileVersion
            self.metadata = metadata
            ys, xs = test_frame.RawImageA.shape
            self.ScanRate = kwargs.get('ScanRate', 1e9/metadata.get('Dwelltime_ns', 100.0))
            self.EHT = kwargs.get('EHT', metadata.get('Landing_Energy_keV', 0))
            self.SEMCurr = kwargs.get('SEMCurr', metadata.get('Beam_Current_pA', 0.0)/1e12)
            self.XResolution = kwargs.get('XResolution', metadata.get('Width', xs))
            self.YResolution = kwargs.get("YResolution", metadata.get('Height', ys))
            self.XResolutions = kwargs.get('XResolutions', np.full(len(fls[0]), self.XResolution))
            self.YResolutions = kwargs.get('YResolutions', np.full(len(fls[0]), self.YResolution))
            self.PixelSize = kwargs.get('PixelSize', metadata.get('Pixelsize_nm', 5.0))
            self.EightBit = kwargs.get('EightBit', 1)
            self.Sample_ID = kwargs.get("Sample_ID",  metadata.get('Experiment', ''))
        self.Scaling = kwargs.get("Scaling", test_frame.Scaling)
        self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        self.DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        self.thr_min = kwargs.get("thr_min", 1e-3)
        self.thr_max = kwargs.get("thr_max", 1e-3)
        self.nbins = kwargs.get("nbins", 256)
        self.sliding_minmax = kwargs.get("sliding_minmax", True)
        self.SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
        self.SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", 3)
        self.SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.025)
        self.SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 10)
        self.SIFT_sigma = kwargs.get("SIFT_sigma", 1.6)
        self.BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        self.save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
        self.TransformType = kwargs.get("TransformType", ShiftTransform)
        l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
        l2_matrix_default = np.eye(6)*l2_param_default             # initially set equal shrinkage on all coefficients
        l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
        l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
        self.l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
        self.targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
        self.solver = kwargs.get("solver", 'RANSAC')
        self.RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.05)  # fraction of data points for initial RANSAC iteration step.
        self.Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)
        self.drmax = kwargs.get("drmax", 1.5)
        self.max_iter = kwargs.get("max_iter", 10000)
        self.SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
        self.save_res_png  = kwargs.get("save_res_png", True)
        self.fnm_types = kwargs.get("fnm_types", ['mrc'])
        self.flipY = kwargs.get("flipY", False)                     # If True, the registered data will be flipped along Y axis
        self.interpolation = kwargs.get("interpolation", cv2.INTER_LINEAR)             #     The order of interpolation. The options are:
                                                                    #    cv2.INTER_AREA    Uses pixel area relation for resampling, which effectively minimizes distortion and avoids aliasing artifacts, yielding high-quality results for reduced image sizes.
                                                                    #    cv2.INTER_CUBIC    Uses bicubic interpolation (based on 4x4 neighboring pixels) to produce smooth, high-quality results. It is slower than INTER_LINEAR.
                                                                    #    cv2.INTER_LINEAR   Uses bilinear interpolation (based on 2x2 neighboring pixels), offering a good balance of speed and visual quality for most general resizing tasks.
                                                                    #    cv2.INTER_NEAREST  Uses the nearest neighbor, picking the value of the closest pixel. It is quick but results in a blocky, pixelated output, and is generally used for specific cases like segmentation masks.
                                                                    #    cv2.INTER_LANCZOS4  Uses a Lanczos kernel with an 8x8 neighborhood, providing the best quality, but it is the slowest method.

        try:
            fnm_mosaic_stack_default = os.path.splitext(os.path.split(fname0)[1])[0][0:-5] + 'mosaic_stack.mrc'
        except:
            fnm_mosaic_stack_default = 'mosaic_stack.mrc'
        self.fnm_mosaic_stack = kwargs.get('fnm_mosaic_stack', fnm_mosaic_stack_default)
        self.dtp = kwargs.get("dtp", np.int16)
        self.nz_tiles = self.fls.shape[0]
        L = self.nz_tiles 
        self.n_tiles_per_layer = len(self.fls[0].ravel())
        kwargs.update({'data_dir' : self.data_dir, 'fnm_mosaic_stack' : self.fnm_mosaic_stack, 'dtp' : self.dtp})

        w_sqrt_intra = np.sqrt(self.intralayer_weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||
        w_sqrt_inter = np.sqrt(self.interlayer_weight)

        if len(image_coordinates_files)>0: # user-defined grid with FirstPixels determined from the image_coordinates_files files        
            FirstPixels = np.zeros((L, self.n_tiles_per_layer, 3))
            for j, fls_layer in enumerate(tqdm(self.fls, desc = 'Reading image coordinates from .txt files', display=verbose)):
                coord_dict = read_image_coordinates(image_coordinates_files[j])
                for i, fl in enumerate(fls_layer.ravel()):
                    p, p1 = os.path.split(fl)
                    tail = '/'.join([os.path.split(p)[-1], p1])
                    FirstPixels[j, i] = coord_dict[tail]
            self.FirstPixels = FirstPixels
            # Find all intra-layer neighbouring pairs by proximity.
            # Two tiles are neighbours if their bounding boxes overlap in both X and Y.
            # This naturally handles hexagonal layouts where each tile has 1 left/right
            # neighbour and up to 2 top/bottom neighbours.
        else:   # standard recti-linear grid with FirstPixels determined from the headers of .dat files
            FirstPixels_layer0 = []
            for fl in tqdm(fls[0].ravel(), desc = 'Reading image coordinates from headers', display=verbose):
                fr = FIBSEM_frame(fl, read_header_only=True)
                FirstPixels_layer0.append([fr.FirstPixelX, fr.FirstPixelY, 0])
            FirstPixels_layer0 = np.array(FirstPixels_layer0)          # shape (n_tiles, 3)
            self.FirstPixels = np.repeat(FirstPixels_layer0[np.newaxis, :, :], L, axis=0)

        # Build intra-layer index pairs — vectorized
        intra_index_pairs_x = []
        intra_index_pairs_y = []
        diag_thresh = self.diagonal_exclusion_threshold * (self.XResolution + self.YResolution)
        for l in tqdm(range(L), desc = 'Determining intra-layer pairs', display=verbose):
            x = self.FirstPixels[l, :, 0]                          # shape (N,)
            y = self.FirstPixels[l, :, 1]
            dx = x[np.newaxis, :] - x[:, np.newaxis]               # dx[i,j] = x[j] - x[i]
            dy = y[np.newaxis, :] - y[:, np.newaxis]
            dx_abs = np.abs(dx)
            dy_abs = np.abs(dy)

            overlap = (dx_abs < self.XResolution) & \
                      (dy_abs < self.YResolution) & \
                      (dx_abs + dy_abs < diag_thresh)
            np.fill_diagonal(overlap, False)

            # X-dominant pairs (dx_abs > dy_abs)
            x_dom = overlap & (dx_abs > dy_abs)
            ii, jj = np.where(x_dom & (dx < 0))                    # j is left of i → store (j, i)
            pairs_x = np.column_stack([jj, ii]) if len(ii) > 0 else np.empty((0, 2), dtype=int)
            if self.add_reverse_edges:
                ii2, jj2 = np.where(x_dom & (dx > 0))
                if len(ii2) > 0:
                    pairs_x = np.vstack([pairs_x, np.column_stack([ii2, jj2])])

            # Y-dominant pairs (dx_abs <= dy_abs)
            y_dom = overlap & (dx_abs <= dy_abs)
            ii, jj = np.where(y_dom & (dy < 0))                    # j is above i → store (j, i)
            pairs_y = np.column_stack([jj, ii]) if len(ii) > 0 else np.empty((0, 2), dtype=int)
            if self.add_reverse_edges:
                ii2, jj2 = np.where(y_dom & (dy > 0))
                if len(ii2) > 0:
                    pairs_y = np.vstack([pairs_y, np.column_stack([ii2, jj2])])

            intra_index_pairs_x.append(pairs_x)
            intra_index_pairs_y.append(pairs_y)

        # Build inter-layer index pairs
        # When coordinates files are provided: full proximity search between adjacent layers.
        # Otherwise: assume same tile index directly above/below.
        inter_index_pairs = []   # list of (L-1) arrays, shape (n_inter_pairs_l, 2)
                                  # columns: (tile_index_in_layer_l, tile_index_in_layer_l+1)
        if len(image_coordinates_files) > 0:
            for l in tqdm(range(L - 1), desc = 'Determining inter-layer pairs', display=verbose):
                x_l  = self.FirstPixels[l,   :, 0]                 # shape (N,)
                y_l  = self.FirstPixels[l,   :, 1]
                x_l1 = self.FirstPixels[l+1, :, 0]
                y_l1 = self.FirstPixels[l+1, :, 1]
                dx_abs = np.abs(x_l1[np.newaxis, :] - x_l[:, np.newaxis])   # [i,j]: layer l tile i vs layer l+1 tile j
                dy_abs = np.abs(y_l1[np.newaxis, :] - y_l[:, np.newaxis])
                ii, jj = np.where((dx_abs < self.XResolution) & (dy_abs < self.YResolution))
                inter_index_pairs.append(np.column_stack([ii, jj]) if len(ii) > 0 else np.empty((0, 2), dtype=int))
        else:
            for l in tqdm(range(L - 1), desc = 'Determining inter-layer pairs', display=verbose):
                inter_index_pairs.append(
                    np.array([(i, i) for i in range(self.n_tiles_per_layer)]))

        nh = sum(len(intra_index_pairs_x[l]) for l in range(L))             # Total number of left-right intra-layer pairs
        self.nh = nh
        nv = sum(len(intra_index_pairs_y[l]) for l in range(L))              # Total number of up-down intra-layer pairs
        self.nv = nv
        n_inter_base = sum(len(inter_index_pairs[l]) for l in range(L - 1))
        nl = n_inter_base * 2 if self.add_reverse_edges else n_inter_base
        self.nl = nl
        self.C = self.nh + self.nv + self.nl
        V = L * self.n_tiles_per_layer                     # Total number of tiles

        # Prepare data for sparse matrix A
        data = []
        row_ind = []
        col_ind = []
        row = 0   # row (entry) in the sparse matrix A (not a tile row)

        # Build a sparse matrix A for Ax=b lsqr equation
        # idx1 and idx2 are absolute (in 1D sense) tile indices
        # each entry is a single sparse matrix element, there are two elements per pairwise translation condition, they enter with opposite signs

        # Intra-layer adjacent pairs
        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Building image pair list')
        for l in tqdm(range(L), desc = 'Adding intra-layer horizontal pairs', display=verbose):
            for i in range(len(intra_index_pairs_x[l])):
                idx1 = l * self.n_tiles_per_layer + intra_index_pairs_x[l][i, 0]
                idx2 = l * self.n_tiles_per_layer + intra_index_pairs_x[l][i, 1]
                row_ind.extend([row, row])
                col_ind.extend([idx1, idx2])
                data.extend([-w_sqrt_intra, w_sqrt_intra])
                row += 1
        for l in tqdm(range(L), desc = 'Adding intra-layer vertical pairs', display=verbose):
            for i in range(len(intra_index_pairs_y[l])):
                idx1 = l * self.n_tiles_per_layer + intra_index_pairs_y[l][i, 0]
                idx2 = l * self.n_tiles_per_layer + intra_index_pairs_y[l][i, 1]
                row_ind.extend([row, row])
                col_ind.extend([idx1, idx2])
                data.extend([-w_sqrt_intra, w_sqrt_intra])
                row += 1
        # Inter-layer adjacent pairs
        for l in tqdm(range(L - 1), desc = 'Adding inter-layer pairs', display=verbose):
            for k in range(len(inter_index_pairs[l])):
                i = inter_index_pairs[l][k, 0]   # tile index in layer l
                j = inter_index_pairs[l][k, 1]   # tile index in layer l+1
                idx1 = l * self.n_tiles_per_layer + i
                idx2 = (l + 1) * self.n_tiles_per_layer + j
                row_ind.extend([row, row])
                col_ind.extend([idx1, idx2])
                data.extend([-w_sqrt_inter, w_sqrt_inter])
                row += 1
                if self.add_reverse_edges:
                    row_ind.extend([row, row])
                    col_ind.extend([idx2, idx1])
                    data.extend([-w_sqrt_inter, w_sqrt_inter])
                    row += 1

        self.index_pairs = np.array(col_ind).reshape((row, 2))   # absolute (in 1D sense) tile indices for each pair
        Xoverlap_per_layer = []
        Yoverlap_per_layer = []
        for l in range(L):
            if len(intra_index_pairs_x[l]) > 0:
                j1, j2 = intra_index_pairs_x[l][0]
                Xoverlap_per_layer.append(int(np.round(
                    self.XResolution - np.abs(self.FirstPixels[l, j1, 0] - self.FirstPixels[l, j2, 0]))))
            else:
                Xoverlap_per_layer.append(0)
            if len(intra_index_pairs_y[l]) > 0:
                i1, i2 = intra_index_pairs_y[l][0]
                Yoverlap_per_layer.append(int(np.round(
                    self.YResolution - np.abs(self.FirstPixels[l, i1, 1] - self.FirstPixels[l, i2, 1]))))
            else:
                Yoverlap_per_layer.append(0)
        # Scalar summaries for backward compatibility
        self.Xoverlap = Xoverlap_per_layer[0] if Xoverlap_per_layer else 0
        self.Yoverlap = Yoverlap_per_layer[0] if Yoverlap_per_layer else 0
        pair_margins = []
        for l in range(L):
            for _ in intra_index_pairs_x[l]:
                pair_margins.append([self.YResolution, 2 * Xoverlap_per_layer[l]])
        for l in range(L):
            for _ in intra_index_pairs_y[l]:
                pair_margins.append([2 * Yoverlap_per_layer[l], self.XResolution])
        for _ in range(self.nl):
            pair_margins.append([self.YResolution, self.XResolution])
        self.pair_margins = pair_margins

        # Per-pair exact overlap rectangles derived from FirstPixels.
        # Replaces the pair_margins assumption that img1 is always left/above img2.
        # Works for rectangular grids and arbitrary layouts equally.
        # Each entry: (x_min_a, x_max_a, y_min_a, y_max_a,
        #              x_min_b, x_max_b, y_min_b, y_max_b)  — all in local pixel coords
        pair_overlap_bounds = []
        for abs_a, abs_b in tqdm(self.index_pairs, desc = 'Determining pair overlap bounds', display=verbose):
            la = int(abs_a) // self.n_tiles_per_layer
            ta = int(abs_a) % self.n_tiles_per_layer
            lb = int(abs_b) // self.n_tiles_per_layer
            tb = int(abs_b) % self.n_tiles_per_layer
            dx = self.FirstPixels[lb, tb, 0] - self.FirstPixels[la, ta, 0]
            dy = self.FirstPixels[lb, tb, 1] - self.FirstPixels[la, ta, 1]
            x_ov = self.XResolution - abs(dx)
            y_ov = self.YResolution - abs(dy)
            x_min_a = max(0,  dx);  x_max_a = x_min_a + x_ov
            y_min_a = max(0,  dy);  y_max_a = y_min_a + y_ov
            x_min_b = max(0, -dx);  x_max_b = x_min_b + x_ov
            y_min_b = max(0, -dy);  y_max_b = y_min_b + y_ov
            pair_overlap_bounds.append((x_min_a, x_max_a, y_min_a, y_max_a,
                                        x_min_b, x_max_b, y_min_b, y_max_b))
        self.pair_overlap_bounds = pair_overlap_bounds

        self.A_csr = csr_matrix((data, (row_ind, col_ind)), shape=(self.C, V)) # sparse matrix

        eye3x3 = np.eye(3,3)
        self.ECC_transformation_matrices = np.repeat(eye3x3[np.newaxis, :, :], self.C, axis=0)
        self.ECC_transformation_valid = np.full(self.C, False)
        self.SIFT_transformation_matrices = np.repeat(eye3x3[np.newaxis, :, :], self.C, axis=0)
        self.SIFT_transformation_valid = np.full(self.C, False)
        self.SIFT_fnms_matches = ['' for x in np.arange(self.C)]
        self.SIFT_nmatches = np.full(self.C, 0)
        self.SIFT_intensity_ratios = np.full(self.C, np.nan)
        self.mean_intensity_ratios   = np.full(self.C, np.nan)
        self.percentile_intensity_ratios = np.full(self.C, np.nan)
        self.overlap_mean_intensity_ratios       = np.full(self.C, np.nan)
        self.overlap_percentile_intensity_ratios = np.full(self.C, np.nan)
        self.overlap_intensity_ratios_valid        = np.full(self.C, False)
        self.target_intensity_ratios        = np.full(self.C, np.nan, dtype=np.float64)
        self.target_intensity_ratios_valid  = np.full(self.C, False)
        self.min_overlap_pixels                  = kwargs.get('min_overlap_pixels', 5000)
        self.percentile = kwargs.get('percentile', 50)
        self.SIFT_Affine_r2norm = np.nan    # residual 2-norm from the affine bundle solve
        self.tile_I0s = np.zeros((self.nz_tiles, self.n_tiles_per_layer))
        self.tile_scales = np.ones((self.nz_tiles, self.n_tiles_per_layer))
        # Pre-init so that `len(self.fnms_kpts) == 0` works in determine_transformations_SIFT
        # before extract_keypoints() has been called. Both are overwritten with shape
        # (nz_tiles, n_tiles_per_layer) numpy arrays inside extract_keypoints (line 2854).
        self.fnms_kpts = np.array([])
        self.nkpts     = np.array([])

        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Initialized FIBSEM_mosaic_dataset instance:')
            print('Total number of tile files: {:d}'.format(V))
            print('Number of tiles per Z-layer: {:d}'.format(self.n_tiles_per_layer))
            print('Number of Z-slices (nz_tiles): {:d}'.format(self.nz_tiles))
            print('')
            print('Total number of left-right intra-layer pairs: ', self.nh)
            print('Total number of up-down intra-layer pairs: ', self.nv)
            print('Total number of intra-layer pairs: ', self.nh + self.nv)
            print('Total number of inter-layer pairs: ', self.nl)
            print('Total number of pairwise transformations : {:d}'.format(self.C))

        self.Xsize = int(np.round(np.max(self.FirstPixels[:, :, 0]) - np.min(self.FirstPixels[:, :, 0]) + self.XResolution))
        self.Ysize = int(np.round(np.max(self.FirstPixels[:, :, 1]) - np.min(self.FirstPixels[:, :, 1]) + self.YResolution))
        
        # Initialize the translation matrix for each tile.
        # tr_matr stores translations as NEGATIVE pixel offsets (tr_matr[:,:,i,2] = -position_i),
        # consistent with the convention that the transformation maps tile-local
        # coordinates to canvas coordinates: x_canvas = x_tile + tr_matr[0,2].
        # tile_positions is no longer stored separately — derive it on demand as
        # -tr_matr[:, :, 0:2, 2] whenever (positive) pixel positions are needed.
        # shifts are per-layer: each layer normalised to its own origin
        shifts_x = self.FirstPixels[:, :, 0] - np.min(self.FirstPixels[:, :, 0], axis=1, keepdims=True)
        shifts_y = self.FirstPixels[:, :, 1] - np.min(self.FirstPixels[:, :, 1], axis=1, keepdims=True)
        # shape of shifts_x / shifts_y: (L, n_tiles_per_layer)

        default_tr_matr = np.broadcast_to(eye3x3, (L, self.n_tiles_per_layer, 3, 3)).copy()
        # shape: (L, n_tiles_per_layer, 3, 3)
        default_tr_matr[:, :, 0, 2] = - shifts_x
        default_tr_matr[:, :, 1, 2] = - shifts_y
        self.default_tr_matr = default_tr_matr
        self.tr_matr = default_tr_matr.copy()   # .copy() so solver writes don't corrupt default

        if memory_profiling:
            elapsed_time = elapsed_since(start_time)
            rss_after, vms_after, shared_after = get_process_memory()
            print("Profiling: Start of Execution: RSS: {:>8} | VMS: {:>8} | SHR {"
                  ":>8} | time: {:>8}"
                .format(format_bytes(rss_after - rss_before),
                        format_bytes(vms_after - vms_before),
                        format_bytes(shared_after - shared_before),
                        elapsed_time))

    def find_tile_pairs(self, layer_id, tile_id, **kwargs):
        '''
        Searches self.index_pairs for all pairs containing the tile.
        Reports SIFT transformation records for each found pair. 
        
        Parameters:
        ----------
        layer_id : int
            Layer ID.
        tile_id : int
            Tile ID within a layer.

        kwargs:
        ----------
        verbose : bool
            If True (default), Verbose output is ON.

        Returns:
        ----------
        res : pandas DataFrame
            Contains these fields: 'Layer 0', 'Tile 0', 'Layer 1', 'Tile 1', 'SIFT x-shift', 'SIFT y-shift', 'SIFT nmatches', 'SIFT valid'
        '''
        verbose = kwargs.get('verbose', True)
        nl = self.n_tiles_per_layer
        abs_tile_id = layer_id * nl + tile_id
        pair_inds = np.argwhere(np.array(self.index_pairs)==abs_tile_id)[:, 0]
        if len(pair_inds) == 0:
            if verbose:
                print('pair_inds contain no record of this tile')
            res = pd.DataFrame(columns=['Layer 0', 'Tile 0', 'Layer 1', 'Tile 1',
                                         'SIFT x-shift', 'SIFT y-shift',
                                         'SIFT nmatches', 'SIFT valid'])
        else:
            tile_pairs = []
            SIFT_shifts = []
            SIFT_valid = []
            SIFT_nmatches = []
            for pair_ind in pair_inds:
                pair = self.index_pairs[pair_ind]
                layer0 = pair[0]//nl
                tile0 = pair[0] - layer0*nl
                layer1 = pair[1]//nl
                tile1 = pair[1] - layer1*nl
                tile_pairs.append([layer0, tile0, layer1, tile1])
                SIFT_shifts.append(self.SIFT_transformation_matrices[pair_ind, 0:2, 2])
                SIFT_valid.append(self.SIFT_transformation_valid[pair_ind])
                SIFT_nmatches.append(self.SIFT_nmatches[pair_ind])
            pd_layers = pd.DataFrame(np.array(tile_pairs), columns = ['Layer 0', 'Tile 0', 'Layer 1', 'Tile 1'])
            pd_SIFT_shifts = pd.DataFrame(np.array(SIFT_shifts), columns = ['SIFT x-shift', 'SIFT y-shift'])
            pd_SIFT_nmatches = pd.DataFrame(np.array(SIFT_nmatches), columns = ['SIFT nmatches'])
            pd_SIFT_valid = pd.DataFrame(np.array(SIFT_valid), columns = ['SIFT valid'])
            res = pd.concat([pd_layers, pd_SIFT_shifts, pd_SIFT_nmatches, pd_SIFT_valid], axis=1)
        if verbose:
            display(res)
        return  res


    def save_parameters(self, **kwargs):
        '''
        Save transformation attributes and parameters (including transformation matrices).

        kwargs:
        -------
        dump_filename : string
            String containing the name of the binary dump for saving all attributes of the current instance of the FIBSEM_dataset object.

        Returns:
        ----------
            dump_filename : string
        '''
        default_dump_filename = os.path.join(self.data_dir, self.fnm_mosaic_stack.replace('.mrc', '_params.bin'))
        dump_filename = kwargs.get("dump_filename", default_dump_filename)

        pickle.dump(self.__dict__, open(dump_filename, 'wb'))

        return dump_filename


    def evaluate_FIBSEM_statistics(self, **kwargs):
        '''
        Evaluates parameters of FIBSEM montage (Min/Max, Working Distance (WD), Milling Y Voltage (MV), FOV center positions). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        max_futures : int
            max number of running DASK futures. Default is self.max_futures (50000).
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        frame_inds : array
            Array of frames to be used for evaluation. If not provided, evaluation will be performed on all frames.
        data_dir : str
            Data directory (path). Default is object attribute.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        percentile : int
            Percentile value for data evaluation. Default is obect attribute (50).
        FIBSEM_Data_parquet : str
            File path of the Parquet file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).
        use_existing_data : boolean
            Default is False. If True and the data exists (saved to Parquet), use that.            
        verbose : boolean
            If True, intermediate messages and results will be displayed. Default is False.

        Returns:
        ----------
        FIBSEM_Data : list of 20 parameters
            FIBSEM_Data_parquet, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions, SEMStiX, SEMStiY, SEMAlnX, SEMAlnY, errors_s2
                FIBSEM_Data_parquet : str
                    path to Parquet file with the FIBSEM data
                data_min_glob : np.float32   
                    min data value for I8 conversion (open CV SIFT requires I8)
                data_max_glob : np.float32   
                    max data value for I8 conversion (open CV SIFT requires I8)
                center_x : np.float32 array
                    FOV Center X-coordinate extracted from the header data
                center_y : np.float32 array
                    FOV Center Y-coordinate extracted from the header data
                ScanRate : np.float32 array
                    SEM Scan Rate (Hz)
                EHT : np.float32 array
                    SEM EHT voltage (kV)
                SEMSpecimenI : np.float32 array
                    SEM Specimen current (nA)
                XResolutions : int array
                    X-frame sizes
                YResolutions : int array
                    Y-frame sizes
        '''
        verbose = kwargs.get('verbose', True)
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
        DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        max_futures = kwargs.get('max_futures', self.max_futures)
        ftype = kwargs.get("ftype", self.ftype)
        frame_inds = kwargs.get("frame_inds", np.arange(self.nz_tiles))
        data_dir = kwargs.get('data_dir', self.data_dir)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        fit_params = kwargs.get('fit_params', ['SG', 3, 1])
        FIBSEM_Data_parquet_default = os.path.join(data_dir, os.path.splitext(self.fnm_mosaic_stack)[0] + '_FIBSEM_Data.parquet')
        FIBSEM_Data_parquet = kwargs.get('FIBSEM_Data_parquet', FIBSEM_Data_parquet_default)
        use_existing_data = kwargs.get('use_existing_data', False)
        if hasattr(self, 'Mill_Volt_Rate_um_per_V'):
            Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", self.Mill_Volt_Rate_um_per_V)
        else:
            Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", 31.235258870176065)
        percentile = kwargs.get('percentile', self.percentile)
        self.percentile = percentile
        local_kwargs = {'use_DASK' : use_DASK,
                        'DASK_client_retries' : DASK_client_retries,
                        'max_futures' : max_futures,
                        'ftype' : ftype,
                        'frame_inds' : np.arange(len(self.fls.ravel())),
                        'data_dir' : data_dir,
                        'thr_min' : thr_min,
                        'thr_max' : thr_max,
                        'nbins' : nbins,
                        'percentile' : percentile,
                        'sliding_minmax' : False,
                        'fit_params' : fit_params,
                        'FIBSEM_Data_parquet' : FIBSEM_Data_parquet,
                        'verbose' : verbose,
                        'use_existing_data' : use_existing_data}

        if verbose:
            print('Evaluating the parameters of FIBSEM data set (data Min/Max, Working Distance, FOV center positions, Scan Rate, EHT)')
        self.FIBSEM_Data = evaluate_FIBSEM_frames_dataset(self.fls.ravel(), DASK_client, **local_kwargs)
        self.data_minmax = [self.FIBSEM_Data['FIBSEM_Data_parquet'],
                    self.FIBSEM_Data['data_min_glob'],
                    self.FIBSEM_Data['data_max_glob'],
                    self.FIBSEM_Data['data_min_sliding'],
                    self.FIBSEM_Data['data_max_sliding']]
        self.data_min_glob  = self.FIBSEM_Data['data_min_glob']
        self.data_max_glob  = self.FIBSEM_Data['data_max_glob']
        WD                  = self.FIBSEM_Data['mill_rate_WD']
        MillingYVoltage     = self.FIBSEM_Data['mill_rate_MV']
        self.FOV_x          = self.FIBSEM_Data['center_x']
        self.FOV_y          = self.FIBSEM_Data['center_y']
        try:
            self.XResolutions   = self.FIBSEM_Data['XResolutions'].astype(int)
            self.YResolutions   = self.FIBSEM_Data['YResolutions'].astype(int)
        except:
            self.XResolutions = np.full(len(WD), self.XResolution).astype(int)
            self.YResolutions = np.full(len(WD), self.YResolution).astype(int)

        self.XResolution = np.max(self.XResolutions)
        self.YResolution = np.max(self.YResolutions)
        
        frame_inds_ext = np.repeat(np.array(frame_inds), self.n_tiles_per_layer)

        try:
            WD_fit_coef = np.polyfit(frame_inds_ext, WD, 1)
            rate_WD = WD_fit_coef[0]*1.0e6
    
            MV_fit_coef = np.polyfit(frame_inds_ext, MillingYVoltage, 1)
            rate_MV = MV_fit_coef[0]*Mill_Volt_Rate_um_per_V*-1.0e3

            Z_pixel_size_WD = rate_WD
            Z_pixel_size_MV = rate_MV

            if ftype == 0:
                if verbose:
                    print('Z pixel = {:.2f} nm  - based on WD data'.format(Z_pixel_size_WD))
                    print('Z pixel = {:.2f} nm  - based on Milling Voltage data'.format(Z_pixel_size_MV))

            # If the milling-rate fit produced a non-positive or non-finite Z spacing
            # (e.g. flat WD trend → slope 0), fall back to PixelSize so downstream
            # OME-NGFF metadata gets a positive scale.
            if not (np.isfinite(Z_pixel_size_WD) and Z_pixel_size_WD > 0):
                if verbose:
                    print(f"  WD-based Z pixel size is {Z_pixel_size_WD}; "
                          f"falling back to PixelSize ({self.PixelSize}) for z spacing.")
                Z_pixel_size_WD = self.PixelSize
            self.voxel_size = np.rec.array(
                (self.PixelSize, self.PixelSize, Z_pixel_size_WD),
                dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        except:
            if verbose:
                print('Could not estimate milling rate')
            self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        if verbose:
            print('Set the voxel size to: ', self.voxel_size)

        return self.FIBSEM_Data
    
    def compute_detector_target_intensity_ratios(self, **kwargs):
        '''
        Compute per-pair "detector-prior" target intensity ratios.

        For each tile, detector ID is parsed from the filename (default pattern
        sfov_NNN). Per-detector average intensity is computed over all tiles
        using that detector, from self.FIBSEM_Data (requires a prior call to
        evaluate_FIBSEM_statistics()). For each pair (a, b),
            T_ab = I_{det(b)} / I_{det(a)}.
        Pairs where either detector ID is unparseable or has < min_tiles_per_detector
        samples are marked invalid and excluded from the prior.

        kwargs:
        -------
        method : 'mean' or 'percentile'.    Default 'percentile'.
        I0 : float.                          Default self.Scaling[1, 0].
        detector_id_pattern : raw str.       Default r'sfov_(\\d+)'.
        min_tiles_per_detector : int.        Default 5.
        verbose : bool.                      Default False.

        Stores:
        -------
        self.detector_ids               : 1D int array, length V (-1 = unknown)
        self.detector_avg_intensities   : dict {detector_id -> avg float}
        self.target_intensity_ratios              : 1D float64, length C
        self.target_intensity_ratios_valid        : 1D bool,   length C
        '''
        import re
        method                 = kwargs.get('method', 'percentile')
        I0                     = kwargs.get('I0', self.Scaling[1, 0])
        pattern                = kwargs.get('detector_id_pattern', r'sfov_(\d+)')
        min_tiles_per_detector = kwargs.get('min_tiles_per_detector', 5)
        verbose                = kwargs.get('verbose', False)

        if not hasattr(self, 'FIBSEM_Data') or self.FIBSEM_Data is None:
            raise RuntimeError("FIBSEM_Data not available. Run evaluate_FIBSEM_statistics() first.")

        if method == 'mean':
            vals = np.array(self.FIBSEM_Data['dmeans'],       dtype=np.float64)
        elif method == 'percentile':
            vals = np.array(self.FIBSEM_Data['dpercentiles'], dtype=np.float64)
        else:
            raise ValueError("method must be 'mean' or 'percentile'.")
        vals_corr = vals - I0    # dark-count corrected

        # --- Parse detector IDs from filenames. ---
        fls_flat = self.fls.ravel()
        rx = re.compile(pattern)
        det_ids = np.full(len(fls_flat), -1, dtype=np.int32)
        for i, fname in enumerate(fls_flat):
            m = rx.search(str(fname))
            if m is not None:
                try:
                    det_ids[i] = int(m.group(1))
                except ValueError:
                    pass
        self.detector_ids = det_ids

        # --- Per-detector averages. ---
        det_avg = {}
        for d in np.unique(det_ids):
            if d < 0:
                continue
            mask = (det_ids == d) & np.isfinite(vals_corr) & (vals_corr > 0)
            if int(np.sum(mask)) >= min_tiles_per_detector:
                det_avg[int(d)] = float(np.mean(vals_corr[mask]))
        self.detector_avg_intensities = det_avg

        if verbose:
            print('Parsed {} unique detector IDs; {} retained (>= {} tiles each).'.format(
                int(np.sum(np.unique(det_ids) >= 0)), len(det_avg), min_tiles_per_detector))

        # --- Per-pair target ratios. ---
        targets = np.full(self.C, np.nan, dtype=np.float64)
        valid   = np.full(self.C, False)
        for k, (abs_a, abs_b) in enumerate(self.index_pairs):
            da = int(det_ids[int(abs_a)])
            db = int(det_ids[int(abs_b)])
            if da in det_avg and db in det_avg:
                ia = det_avg[da]
                ib = det_avg[db]
                if ia > 0 and ib > 0:
                    targets[k] = ib / ia
                    valid[k]   = True
        self.target_intensity_ratios       = targets
        self.target_intensity_ratios_valid = valid

        if verbose:
            print('Target ratios: {:d}/{:d} pairs valid, range [{:.4f}, {:.4f}]'.format(
                int(np.sum(valid)), self.C,
                float(np.min(targets[valid])) if np.any(valid) else float('nan'),
                float(np.max(targets[valid])) if np.any(valid) else float('nan')))
        return targets


    def compute_solved_pair_overlap_bounds(self):
        '''
        Compute fresh per-pair overlap rectangles from the CURRENT (post-solve)
        translations in self.tr_matr, instead of the init-time FirstPixels.
        Vectorized over all C pairs in one numpy pass. (c)G.Shtengel gleb.shtengel@gmail.com

        Only the translation component is used; affine scale/rotation in tr_matr
        is ignored (acceptable for ShiftTransform and a good approximation for
        near-translation AffineTransform solves).

        Returns
        -------
        bounds : np.ndarray of shape (C, 8), dtype int64
            Per-pair overlap rectangle columns:
            (x_min_a, x_max_a, y_min_a, y_max_a, x_min_b, x_max_b, y_min_b, y_max_b)
            All zeros for pairs with no overlap.
        areas  : np.ndarray of shape (C,), dtype int64
            Number of overlap pixels per pair (0 when there is no overlap).

        Side effects
        ------------
        Stores results as self.solved_pair_overlap_bounds (now an ndarray, was a list)
        and self.pair_overlap_areas.
        '''
        positions = -self.tr_matr[:, :, 0:2, 2]        # (nz_tiles, n_tiles_per_layer, 2)
        positions_flat = positions.reshape(-1, 2)      # (total_tiles, 2)
        abs_a = self.index_pairs[:, 0].astype(int)
        abs_b = self.index_pairs[:, 1].astype(int)
        dx = positions_flat[abs_b, 0] - positions_flat[abs_a, 0]   # (C,)
        dy = positions_flat[abs_b, 1] - positions_flat[abs_a, 1]
        x_ov = (self.XResolution - np.abs(dx))                     # (C,)
        y_ov = (self.YResolution - np.abs(dy))
        valid = (x_ov > 0) & (y_ov > 0)
        x_ov = np.where(valid, x_ov, 0.0).astype(np.int64)
        y_ov = np.where(valid, y_ov, 0.0).astype(np.int64)
        x_min_a = np.where(valid, np.maximum(0,  dx), 0).astype(np.int64)
        y_min_a = np.where(valid, np.maximum(0,  dy), 0).astype(np.int64)
        x_min_b = np.where(valid, np.maximum(0, -dx), 0).astype(np.int64)
        y_min_b = np.where(valid, np.maximum(0, -dy), 0).astype(np.int64)
        bounds = np.stack([x_min_a, x_min_a + x_ov, y_min_a, y_min_a + y_ov,
                           x_min_b, x_min_b + x_ov, y_min_b, y_min_b + y_ov], axis=1)
        areas = x_ov * y_ov
        self.solved_pair_overlap_bounds = bounds
        self.pair_overlap_areas         = areas
        return bounds, areas


    def compute_frame_intensity_ratios(self, **kwargs):
        '''
        Compute per-pair intensity ratios from whole-frame mean or median pixel values.
        Requires evaluate_FIBSEM_statistics() to have been called so that self.FIBSEM_Data
        contains 'dmeans' and 'dpercentiles' arrays (one value per tile, indexed by flat tile index).

        kwargs:
        ----------
        method : str
            'mean' or 'percentile'. Default is 'percentile'.
        I0 : float
            Dark-count offset subtracted before computing ratios. Default is self.Scaling[1, 0].
        verbose : boolean
            Display summary statistics. Default is False.

        Returns:
        ----------
        ratios : 1D np.float64 array, shape (C,)
            Per-pair intensity ratio dst/src. Stored as self.mean_intensity_ratios
            or self.percentile_intensity_ratios depending on method.
        '''
        method = kwargs.get('method', 'percentile')
        I0 = kwargs.get('I0', self.Scaling[1, 0])
        verbose = kwargs.get('verbose', False)

        if not hasattr(self, 'FIBSEM_Data') or self.FIBSEM_Data is None:
            raise RuntimeError("FIBSEM_Data not available. Run evaluate_FIBSEM_statistics() first.")

        if method == 'mean':
            vals = np.array(self.FIBSEM_Data['dmeans'], dtype=np.float64)
            attr_name = 'mean_intensity_ratios'
        elif method == 'percentile':
            vals = np.array(self.FIBSEM_Data['dpercentiles'], dtype=np.float64)
            attr_name = 'percentile_intensity_ratios'
        else:
            raise ValueError("method '{}' not supported. Use 'mean' or 'percentile'.".format(method))

        vals_corr = vals - I0
        tile_i = self.index_pairs[:, 0]
        tile_j = self.index_pairs[:, 1]
        vi = vals_corr[tile_i]
        vj = vals_corr[tile_j]
        ratios = np.where((vi > 0) & (vj > 0), vj / vi, np.nan)

        setattr(self, attr_name, ratios)
        if verbose:
            valid = np.isfinite(ratios) & (ratios > 0)
            label = '{} (p={})'.format(method, self.percentile) if method == 'percentile' else method
            print('{} intensity ratios: {:d}/{:d} valid pairs, mean={:.4f}, std={:.4f}'.format(
                label, int(np.sum(valid)), self.C,
                float(np.mean(ratios[valid])) if np.any(valid) else float('nan'),
                float(np.std(ratios[valid]))  if np.any(valid) else float('nan')))
        return ratios


    def compute_overlap_intensity_ratios(self, **kwargs):
        '''
        Compute per-pair intensity ratios from the MEAN or PERCENTILE intensity
        inside each pair's overlap ROI, using the post-solve tile positions
        (self.tr_matr). Parallelized across TILES using DASK so that each tile
        file is read at most once even when it participates in several pairs.
        ©G.Shtengel gleb.shtengel@gmail.com

        kwargs
        ------
        method : str
            'mean' or 'percentile'. Default 'percentile'.
        I0 : float
            Dark-count offset subtracted before computing ratios.
            Default self.Scaling[1, 0].
        percentile : int
            Percentile used when method == 'percentile'. Default self.percentile.
        min_overlap_pixels : int
            Pairs with overlap area below this threshold are marked invalid
            (analogous to SIFT_nmatches_min). Default self.min_overlap_pixels (5000).
        DASK_client : DASK client or '' for local. Default ''.
        DASK_client_retries : int. Default self.DASK_client_retries (or 3).
        ftype : int. Default self.ftype.
        verbose : bool. Default False.
        max_futures : int
            Max number of running DASK futures per batch. Default is self.max_futures (50000).
            Reduces scheduler load by submitting in waves. Each wave's gather completes
            before the next is submitted.

        Returns
        -------
        ratios : 1D np.float64, shape (C,)
            Per-pair intensity ratio dst/src. Stored as
            self.overlap_mean_intensity_ratios or self.overlap_percentile_intensity_ratios.

        Side effects
        ------------
        Updates self.overlap_intensity_ratios_valid (True iff overlap area >=
        min_overlap_pixels AND both mean intensities are positive after I0 subtraction).
        '''
        method             = kwargs.get('method', 'percentile')
        I0                 = kwargs.get('I0', self.Scaling[1, 0])
        percentile         = kwargs.get('percentile', self.percentile)
        min_overlap_pixels = kwargs.get('min_overlap_pixels', self.min_overlap_pixels)
        DASK_client        = kwargs.get('DASK_client', '')
        DASK_client_retries = kwargs.get('DASK_client_retries',
                                         getattr(self, 'DASK_client_retries', 3))
        ftype              = kwargs.get('ftype', self.ftype)
        verbose            = kwargs.get('verbose', False)

        if method not in ('mean', 'percentile'):
            raise ValueError("method '{}' not supported. Use 'mean' or 'percentile'.".format(method))

        use_DASK, _ = check_DASK(DASK_client, verbose=True)

        bounds, areas = self.compute_solved_pair_overlap_bounds()
        pair_valid_by_area = areas >= min_overlap_pixels

        # --- Build per-tile ROI work lists. ---
        # roi_id = (pair_index, side)  with side in (0, 1)  for (a, b).
        per_tile_rois = {}
        for j, (abs_a, abs_b) in enumerate(self.index_pairs):
            if not pair_valid_by_area[j]:
                continue
            x_min_a, x_max_a, y_min_a, y_max_a, x_min_b, x_max_b, y_min_b, y_max_b = bounds[j]
            per_tile_rois.setdefault(int(abs_a), []).append(((j, 0), x_min_a, x_max_a, y_min_a, y_max_a))
            per_tile_rois.setdefault(int(abs_b), []).append(((j, 1), x_min_b, x_max_b, y_min_b, y_max_b))

        fls_flat = self.fls.ravel()
        worker_kwargs = {'ftype': ftype, 'method': method, 'percentile': percentile}
        params = [[fls_flat[t], rois, worker_kwargs] for t, rois in per_tile_rois.items()]

        # --- Dispatch (mirrors evaluate_FIBSEM_frames_dataset). ---
        max_futures = kwargs.get('max_futures', self.max_futures)
        results = []
        if len(params) > 0:
            if use_DASK:
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Using DASK distributed for overlap intensity computation')
                n_tasks   = len(params)
                n_batches = (n_tasks + max_futures - 1) // max_futures
                for DASK_batch in tqdm(range(n_batches), desc='compute_overlap_intensity_ratios DASK batches'):
                    start = DASK_batch * max_futures
                    stop  = min(start + max_futures, n_tasks)
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')
                              + '   Starting DASK batch {:d}/{:d} with {:d} jobs ({:d} remaining after this batch)'.format(
                                    DASK_batch + 1, n_batches, stop - start, n_tasks - stop))
                    futures = DASK_client.map(compute_tile_overlap_intensities, params[start:stop],
                                              retries=DASK_client_retries)
                    results += DASK_client.gather(futures)
            else:
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Using Local Computation for overlap intensity computation')
                for p in tqdm(params, desc='Computing overlap intensities', display=verbose):
                    results.append(compute_tile_overlap_intensities(p))

        # --- Merge per-tile result dicts. ---
        roi_vals = {}
        for r in results:
            roi_vals.update(r)

        # --- Assemble per-pair ratios + validity. ---
        ratios = np.full(self.C, np.nan, dtype=np.float64)
        valid  = np.full(self.C, False)
        for j in range(self.C):
            if not pair_valid_by_area[j]:
                continue
            va = roi_vals.get((j, 0), np.nan)
            vb = roi_vals.get((j, 1), np.nan)
            if not (np.isfinite(va) and np.isfinite(vb)):
                continue
            vi = va - I0
            vj = vb - I0
            if vi > 0 and vj > 0:
                ratios[j] = vj / vi
                valid[j]  = True

        attr_name = 'overlap_' + method + '_intensity_ratios'
        setattr(self, attr_name, ratios)
        self.overlap_intensity_ratios_valid = valid

        if verbose:
            label = '{} (p={})'.format(method, percentile) if method == 'percentile' else method
            print('{} overlap ratios: {:d}/{:d} valid pairs (area >= {:d} px), mean={:.4f}, std={:.4f}'.format(
                label, int(np.sum(valid)), self.C, int(min_overlap_pixels),
                float(np.mean(ratios[valid])) if np.any(valid) else float('nan'),
                float(np.std(ratios[valid]))  if np.any(valid) else float('nan')))
        return ratios


    def extract_keypoints(self, **kwargs):
        '''
        Extract Key-Points and Descriptors. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        DASK_client : DASK client. If empty string '' (Default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        max_futures : int
            Max number of running DASK futures per batch. Default is self.max_futures (50000).
            Reduces scheduler load by submitting in waves. Each wave's gather completes
            before the next is submitted.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        EightBit : int
            0 - 16-bit data, 1: 8-bit data. Default is object attribute.
        data_dir : str
            Data directory (path). Default is object attribute.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        left_crop : int
            left image margin to be cropped off BEFORE distortion correction (via deformation field) is applied. Default is object attribute (or 0 if absent).
        deformation_field : 3D array
            Array with dimensions (YResolution, XResolution - left_crop, 2). Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
            Deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        U8_conversion : str
            Range selection for U8 conversion. Options are: 'global', 'sliding', and 'local'. Default is 'local'.
        data_minmax : list of 5 parameters
            minmax_parquet : str
                path to Parquet file with Min/Max data.
            data_min_glob : np.float32   
                min data value for I8 conversion (open CV SIFT requires I8).
            data_min_sliding : np.float32 array
                min data values (one per file) for I8 conversion.
            data_max_sliding : np.float32 array
                max data values (one per file) for I8 conversion.
            data_minmax_glob : 2D np.float32 array
                min and max data values without sliding averaging.
        SIFT_nfeatures : int
            The number of best features to retain. Default is object attribute. SIFT library default is 0 (all features retained).
            The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        SIFT_nOctaveLayers : int
            The number of layers in each octave. Default is object attribute. SIFT library default is 3.
            3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        SIFT_contrastThreshold : double
            The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. Default is object attribute. SIFT library default is 0.04.
            The larger the threshold, the less features are produced by the detector.
            The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
            When nOctaveLayers is set to default and if you want to use the value used in
            D. Lowe paper (0.03), set this argument to 0.09.
        SIFT_edgeThreshold : double
            The threshold used to filter out edge-like features. Default is object attribute. SIFT library default is 10.
            Note that its meaning is different from the contrastThreshold,
            i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        SIFT_sigma : double
            The sigma of the Gaussian applied to the input image at the octave #0. Default is object attribute. SIFT library default is 1.6.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.    
        interpolation : int
            Interpolation type as defined in CV2. Default is object attribute (default for that is cv2.INTER_LINEAR).
        fill_value : float
            Fill value for outside pixels in cv2.remap. Default is 0.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
            If True, outputs will be printed.
    
        Returns:
        ----------
        fnms_kpts : array of str
            Filenames for binary files containing Key-Points and Descriptors for each frame.
        '''
        verbose = kwargs.get('verbose', True)
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
        DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        ftype = kwargs.get("ftype", self.ftype)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        U8_conversion = kwargs.get('U8_conversion', self.U8_conversion)
        if U8_conversion != 'local':
            data_minmax = kwargs.get("data_minmax", self.data_minmax)
            minmax_parquet, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        left_crop = kwargs.get('left_crop', self.left_crop)
        deformation_field = kwargs.get('deformation_field', self.deformation_field)
        interpolation = kwargs.get('interpolation', self.interpolation)
        fill_value = kwargs.get('fill_value', 0)
        use_existing_data = kwargs.get('use_existing_data', False)
        
        kpt_kwargs = {'ftype' : ftype,
                    'thr_min' : thr_min,
                    'thr_max' : thr_max,
                    'nbins' : nbins,
                    'SIFT_nfeatures' : SIFT_nfeatures,
                    'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
                    'SIFT_contrastThreshold' : SIFT_contrastThreshold,
                    'SIFT_edgeThreshold' : SIFT_edgeThreshold,
                    'SIFT_sigma' : SIFT_sigma,
                    'use_existing_data' : use_existing_data,
                    'interpolation' : interpolation,
                    'fill_value' : fill_value,
                    'left_crop' : left_crop}

        if U8_conversion == 'sliding':
            params_s3 = []
            for j, fl in enumerate(self.fls.ravel()):
                params_s3.append([fl, data_min_sliding[j], data_max_sliding[j], kpt_kwargs])
        else:
            if U8_conversion == 'global': 
                params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls.ravel()]
            else:
                params_s3 = [[fl, -1, -1, kpt_kwargs] for fl in self.fls.ravel()]
  
        max_futures = kwargs.get('max_futures', self.max_futures)
        if use_DASK:
            # Scatter deformation_field once — all tasks reference it via the future.
            shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
            # Stage submissions to avoid choking the scheduler on million-task jobs.
            results_s3 = []
            n_tasks   = len(params_s3)
            n_batches = (n_tasks + max_futures - 1) // max_futures
            for DASK_batch in tqdm(range(n_batches), desc='extract_keypoints DASK batches'):
                start = DASK_batch * max_futures
                stop  = min(start + max_futures, n_tasks)
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Starting DASK batch {:d}/{:d} with {:d} jobs ({:d} remaining after this batch)'.format(
                                DASK_batch + 1, n_batches, stop - start, n_tasks - stop))
                futures_s3 = DASK_client.map(extract_keypoints_descr_files,
                                             params_s3[start:stop],
                                             deformation_field=shared_data_future,
                                             retries=DASK_client_retries)
                results_s3 += DASK_client.gather(futures_s3)
        else:
            results_s3 = []
            for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ', display=verbose)):
                results_s3.append(extract_keypoints_descr_files(param_s3, deformation_field))
        fnms_kpts = [r[0] for r in results_s3]
        nkpts = [r[1] for r in results_s3]
        self.fnms_kpts = np.array(fnms_kpts).reshape(self.fls.shape)
        self.nkpts = np.array(nkpts).reshape(self.fls.shape)
        return fnms_kpts, nkpts

    def analyze_kpt_statistics(self, **kwargs):
        '''
        Analyze key-point statistic and report suspect outliers. ©G.Shtengel 04/2026 gleb.shtengel@gmail.com
        
        Parameters:
        ----------

        kwargs:
        ----------
        sigma_thr : float
            Threshold (multiplied by sigma) for outlier determination. Default is 6.0 (6-sigma outliers).
        save_png : boolean
            If True (default), the plot is saved into PNG file.
        dpi : int
            DPI for PNG. Default is 300.
        save_fname : string
            File name to save the PNG image. Default is os.path.join(data_dir, 'nkpts_Outliers.png').
        verbose : boolean
            Display intermediate results. Default is False.
        mark_outliers : boolean
            If True (default), each outlier is marked with "x" and its frame and tile number are printed next to "x".

        Returns:
        ----------
        outliers : np.ndarray, shape (N, 2)
            Sorted array of [frame_index, tile_index] for all outlier detections.
            Returns empty array of shape (0, 2) if no outliers are found.
        
        '''
        sigma_thr = kwargs.get('sigma_thr', 6.0)
        nxny = self.n_tiles_per_layer
        data_dir = kwargs.get("data_dir", self.data_dir)
        verbose = kwargs.get('verbose', False)
        save_png = kwargs.get('save_png', True)
        dpi = kwargs.get('dpi', 300)
        mark_outliers = kwargs.get('mark_outliers', True)
        fsmark = 6
        if save_png:
            save_fname = kwargs.get('save_fname', os.path.join(data_dir, 'Nkpts_Outliers.png'))
        else:
            save_fname = 'Image not saved'
        if verbose:
            print('Loading kwarg Data')
        Sample_ID = kwargs.get('Sample_ID', '')
        fit_params = kwargs.get("fit_params", ['SG', 11, 3])
        frames = np.arange(self.nz_tiles)
        if verbose:
            print('Generating Plots')
        fs = 12
        fig, ax = plt.subplots(1,1, figsize = (6,4), sharex=True)
        fig.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.05)
        if fit_params[0] != 'None':
            sv_apert = min([fit_params[1], len(frames)//8*2+1])
            if verbose:
                print('Using fit_params: ', 'SG', sv_apert, fit_params[2])
            
        outliers_nkpts = []
        for k in np.arange(nxny):
            my_col = plt.get_cmap("gist_rainbow_r")(0.0 if nxny == 1 else (nxny-k)/(nxny-1))
            tilek_nkpts = self.nkpts[:, k]
            ax.plot(frames, tilek_nkpts, color=my_col, linewidth=0.5)
            if fit_params[0] != 'None':
                sliding_tilek_nkpts = savgol_filter(tilek_nkpts.astype(np.double), sv_apert, fit_params[2])
            else:
                sliding_tilek_nkpts = np.full_like(tilek_nkpts, np.mean(tilek_nkpts), dtype=np.double)
            tilek_nkpts_delta = tilek_nkpts - sliding_tilek_nkpts
            tilek_nkpts_std = np.std(tilek_nkpts_delta)
            outliers_tilek_nkpts = np.where(np.abs(tilek_nkpts_delta) > tilek_nkpts_std * sigma_thr)[0]
            if len(outliers_tilek_nkpts) > 0:
                outliers_nkpts.append([k, outliers_tilek_nkpts])
            if mark_outliers:
                ax.plot(frames[outliers_tilek_nkpts], tilek_nkpts[outliers_tilek_nkpts], color=my_col, marker='x', markersize=4, linestyle='')
                for outlier_tilek_nkpts in outliers_tilek_nkpts:
                    ax.text(frames[outlier_tilek_nkpts], tilek_nkpts[outlier_tilek_nkpts], '{:d}, {:d}'.format(k, frames[outlier_tilek_nkpts]), fontsize=fsmark)
        outliers = reformat_outliers_data(outliers_nkpts)
        ax.set_ylabel('# of Key-Points')
        ax.set_xlabel('Frame')
        ax.text(0.2, 1.04, Sample_ID, fontsize = fs, transform=ax.transAxes)
        ax.grid(True)
        if save_png:
            ax.text(-0.12, -0.17, save_fname, fontsize=5, transform=ax.transAxes)
            fig.savefig(save_fname, dpi=dpi)
        display(fig)
        plt.close(fig)
        return outliers


    def get_interlayer_pairs_mask(self, tile_indices):
        '''
        Return the subset of self.index_pairs that are inter-layer pairs
        for the specified intra-layer tile positions. ©G.Shtengel 04/2026 gleb.shtengel@gmail.com

        Parameters:
        -----------
        tile_indices : array-like of int
            Intra-layer tile indices (0 to n_tiles_per_layer-1) to include.

        Returns:
        --------
        mask : 1D np.ndarray of bool, shape (N_pairs,)
            Boolean mask over self.index_pairs selecting inter-layer pairs
            for the specified intra-layer tile positions.
        '''
        tile_indices = np.asarray(tile_indices)
        n = self.n_tiles_per_layer
        pairs  = self.index_pairs            # shape (N_pairs, 2)
        layer1 = pairs[:, 0] // n
        layer2 = pairs[:, 1] // n
        tile  = pairs[:, 0] % n
        mask = (layer1 != layer2) & np.isin(tile, tile_indices)
        return mask
    

    def determine_transformations_SIFT(self, **kwargs):
        '''
        Determine transformation matrices for frame pairs using SIFT. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        max_futures : int
            Max number of running DASK futures per batch. Default is self.max_futures (50000).
            Reduces scheduler load by submitting in waves. Each wave's gather completes
            before the next is submitted.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_field or on its own). Default is 0 - no cropping.
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is np.nan - no distortion correction.
        select_tiles : list 
            List of tile ID's. Sefault is False or empty list.
        TransformType : object reference
            Transformation model used for determining the transformation matrix from Key-Point pairs. Default is object attribute.
            Choose from the following options:
                ShiftTransform - only x-shift and y-shift
                RotationShiftTransform - x-shift, y-shift, rotation
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D np.float32 array
           Matrix of regularization (shrinkage) parameters (applicable only if RegularizedAffineTransform is used). Default is object attribute.
        targ_vector : 1D np.float32 array
            Target vector for regularization (applicable only if RegularizedAffineTransform is used). Default is object attribute.
        solver : str
            Solver used for SIFT ('RANSAC' or 'LinReg'). Default is object attribute.
        RANSAC_initial_fraction : float
            Fraction of data points for initial RANSAC iteration step. Default is object attribute.
        Lowe_Ratio_Threshold : float
            Threshold for Lowe's Ratio Test. Default is object attribute.
        BFMatcher : boolean
            If True, the BF Matcher is used for Key-Point matching, otherwise FLANN will be used. Default is object attribute.
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression.
            Default is object attribute.
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg). Default is object attribute.
        SIFT_nmatches_min : int
            Min number of matches for the transformation to be considered valid. Default is 5.
        save_matches : boolean
            If True, matches will be saved into individual files. Default is object attribute.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is object attribute.
        start : string
            Start of search for determining FWHM of the error distributions. Options are 'edges' (default) or 'center'.
        estimation : string
            Returns a width of interval determined using search direction from above or total number of bins above half max. Options are 'interval' (default) or 'count'.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
            Display intermediate results. Default is True.
    
        Returns:
        ----------
        transformations_results_3D : array of lists containing the results:
            [transformation_matrix, fnm_matches, npt, kpt_ints, error_abs_mean, error_FWHMx, error_FWHMy, iteration]
            transformation_matrix : 2D np.float32 array
                Transformation matrix for each sequential frame pair.
            fnm_matches : str
                Filename containing the matches used to determine the transformation for the pair of frames.
            npts : int
                Number of matches.
            kpt_ints : list of keypoint intensities
            error_abs_mean : float
                Mean abs error of registration for all matched Key-Points.
        '''
        verbose = kwargs.get('verbose', False)

        if len(self.fnms_kpts) == 0:
            if verbose:
                print('No data on individual key-point data files, perform key-point search')
            transformations_results = []
        else:
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
            select_tiles = kwargs.get('select_tiles', False)
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            ftype = kwargs.get("ftype", self.ftype)
            left_crop = kwargs.get('left_crop', self.left_crop)
            deformation_field = kwargs.get('deformation_field', self.deformation_field)
            TransformType = kwargs.get("TransformType", self.TransformType)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
            Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            save_res_png  = kwargs.get("save_res_png", self.save_res_png )
            start = kwargs.get('start', 'edges')
            estimation = kwargs.get('estimation', 'interval')
            use_existing_data = kwargs.get('use_existing_data', False)

            params_SIFT = []

            if select_tiles:
                mask = self.get_interlayer_pairs_mask(select_tiles)
                index_pairs = self.index_pairs[mask]
                pair_overlap_bounds = [self.pair_overlap_bounds[i] for i in np.where(mask)[0]]
            else:
                index_pairs = self.index_pairs
                pair_overlap_bounds = kwargs.get('pair_overlap_bounds', self.pair_overlap_bounds)
            fnms_kpts = self.fnms_kpts.ravel()

            for (jj, index_pair), overlap_bounds in zip(enumerate(tqdm(index_pairs, desc='Setting up SIFT parameter list', display=verbose)), pair_overlap_bounds):
                dt_kwargs = {'ftype' : ftype,
                        'TransformType' : TransformType,
                        'l2_matrix' : l2_matrix,
                        'targ_vector': targ_vector, 
                        'solver' : solver,
                        'RANSAC_initial_fraction' : RANSAC_initial_fraction,
                        'drmax' : drmax,
                        'max_iter' : max_iter,
                        'BFMatcher' : BFMatcher,
                        'save_matches' : save_matches,
                        'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
                        'start' : start,
                        'estimation' : estimation,
                        'use_existing_data' : use_existing_data,
                        'verbose' : verbose,
                        'left_crop' : left_crop}

                fname1 = fnms_kpts[index_pair[0]]
                fname2 = fnms_kpts[index_pair[1]]
                path_base, f1 = os.path.split(fname1)
                _, f2 = os.path.split(fname2)
                if select_tiles:
                    fnm_matches = os.path.join(path_base, 'pair{:d}_'.format(jj) + f1.replace('_kpdes.bin', '_')+f2.replace('_kpdes.bin', '_select_tile_matches.bin'))
                else:
                    fnm_matches = os.path.join(path_base, 'pair{:d}_'.format(jj) + f1.replace('_kpdes.bin', '_')+f2.replace('_kpdes.bin', '_matches.bin'))
                dt_kwargs['fnm_matches'] = fnm_matches
                dt_kwargs['overlap_bounds'] = overlap_bounds   # (x_min_a, x_max_a, y_min_a, y_max_a,
                                                                #  x_min_b, x_max_b, y_min_b, y_max_b)
                dt_kwargs['image_shape'] = (self.YResolution, self.XResolution)
                la, ta = int(index_pair[0]) // self.n_tiles_per_layer, int(index_pair[0]) % self.n_tiles_per_layer
                lb, tb = int(index_pair[1]) // self.n_tiles_per_layer, int(index_pair[1]) % self.n_tiles_per_layer
                dx = self.FirstPixels[lb, tb, 0] - self.FirstPixels[la, ta, 0]
                dy = self.FirstPixels[lb, tb, 1] - self.FirstPixels[la, ta, 1]
                dt_kwargs['warp_matrix'] = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
                param_SIFT = [fname1, fname2, dt_kwargs]
                params_SIFT.append(param_SIFT)
                if verbose:
                    print('Added a set: ')
                    print([fname1, fname2, dt_kwargs])

            max_futures = kwargs.get('max_futures', self.max_futures)
            if use_DASK:
                # Stage submissions to avoid choking the scheduler on million-pair jobs.
                transformations_results_3D = []
                n_tasks   = len(params_SIFT)
                n_batches = (n_tasks + max_futures - 1) // max_futures
                for DASK_batch in tqdm(range(n_batches), desc='determine_transformations_SIFT DASK batches'):
                    start = DASK_batch * max_futures
                    stop  = min(start + max_futures, n_tasks)
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')
                              + '   Starting DASK batch {:d}/{:d} with {:d} jobs ({:d} remaining after this batch)'.format(
                                    DASK_batch + 1, n_batches, stop - start, n_tasks - stop))
                    futures_SIFT = DASK_client.map(determine_transformations_files,
                                                   params_SIFT[start:stop],
                                                   retries=DASK_client_retries)
                    transformations_results_3D += DASK_client.gather(futures_SIFT)
            else:
                transformations_results_3D = []
                for param_SIFT in tqdm(params_SIFT, desc = 'Extracting Transformation Parameters: ', display=verbose):
                    transformations_results_3D.append(determine_transformations_files(param_SIFT))
            
            if select_tiles:
                self.select_SIFT_transformation_matrices = np.array([np.nan_to_num(transformations_result[0]) for transformations_result in transformations_results_3D]).reshape((self.nz_tiles - 1, len(select_tiles), 3, 3))
            else:
                for j, transformations_result  in enumerate(tqdm(transformations_results_3D, desc = 'Parsing the SIFT results', display = verbose)):
                    try:
                        self.SIFT_transformation_matrices[j] = np.nan_to_num(transformations_result[0])
                        self.SIFT_fnms_matches[j] = transformations_result[1]
                        self.SIFT_nmatches[j] = len(transformations_result[2][0])
                        self.SIFT_transformation_valid[j] = self.SIFT_nmatches[j] > SIFT_nmatches_min
                        src_selected_ints, dst_selected_ints = transformations_result[3]
                        self.SIFT_intensity_ratios[j] = np.mean(dst_selected_ints / src_selected_ints)
                    except Exception as e:
                        if verbose:
                            print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   An error occurred: {}'.format(e))
                            print('transformations_result:  ', transformations_result)
            
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for intra-layer horisontal matches :', np.mean(self.SIFT_nmatches[0:self.nh]).astype(np.int64))
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for intra-layer vertical matches :', np.mean(self.SIFT_nmatches[self.nh:self.nh+self.nv]).astype(np.int64))
                if self.nl > 0:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for inter-layer matches :', np.mean(self.SIFT_nmatches[self.nh+self.nv:]).astype(np.int64))
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   {:d} out of {:d} SIFT transformations are valid  (SIFT_nmatches > {:d})'.format(np.sum(self.SIFT_transformation_valid), self.C, SIFT_nmatches_min))
        return transformations_results_3D


    def SIFT_evaluation(self, index_pair, **kwargs):
        '''
        Evaluate SIFT performance on a given index_pair. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        Parameters:
        index_pair : tuple of 2 ints
            Pair of absolute (in 1D sense of fls.ravel()) tile indices.

        kwargs:
        ----------
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        left_crop : int
            left image margin to be cropped off BEFORE distortion correction (via deformation field) is applied. Default is object attribute (or 0 if absent).
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is image attribute (or np.nan if absent - no distortion correction).
            Deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        data_minmax : list of 5 parameters
            minmax_parquet : str
                path to Parquet file with Min/Max data.
            data_min_glob : np.float32   
                min data value for I8 conversion (open CV SIFT requires I8).
            data_min_sliding : np.float32 array
                min data values (one per file) for I8 conversion.
            data_max_sliding : np.float32 array
                max data values (one per file) for I8 conversion.
            data_minmax_glob : 2D np.float32 array
                min and max data values without sliding averaging.
        I0 : float
            Dark Count used for Intensity calculation. Default is self.Scaling[1,0].
        SIFT_nfeatures : int
            The number of best features to retain. Default is object attribute. SIFT library default is 0 (all features retained).
            The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        SIFT_nOctaveLayers : int
            The number of layers in each octave. Default is object attribute. SIFT library default is 3.
            3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        SIFT_contrastThreshold : double
            The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. Default is object attribute. SIFT library default is 0.04.
            The larger the threshold, the less features are produced by the detector.
            The contrast threshold will be divided by nOctaveLayers when the filtering is applied.
            When nOctaveLayers is set to default and if you want to use the value used in
            D. Lowe paper (0.03), set this argument to 0.09.
        SIFT_edgeThreshold : double
            The threshold used to filter out edge-like features. Default is object attribute. SIFT library default is 10.
            Note that its meaning is different from the contrastThreshold,
            i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        SIFT_sigma : double
            The sigma of the Gaussian applied to the input image at the octave #0. Default is object attribute. SIFT library default is 1.6.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        TransformType : object reference
            Transformation model used for determining the transformation matrix from Key-Point pairs. Default is object attribute.
            Choose from the following options:
                ShiftTransform - only x-shift and y-shift
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        interpolation : int
            Interpolation type as defined in CV2. Default is object attribute (default for that is cv2.INTER_LINEAR).
        fill_value : float
            Fill value for outside pixels in cv2.remap. Default is 0.
        save_res_png : boolean
            If True (Default), the results are saved into a PNG file.
        save_filename : str
            A path for saving PNG data. Default is auto-generated as os.path.join(self.data_dir, os.path.split(fnm_matches)[1].replace('_matches.bin', '_SIFT_test.png').
       
        Returns:
        ----------
        fnm_deformed1, fnm_deformed2, transformations_result, int_results
            int_results is pd.Dataframe with columns: ['X-src', 'Y-src', 'X-src transformed', 'Y-src transformed', 'X-dst', 'Y-dst', 'X-error', 'Y-error', 'Int-src', 'Int-dst']
        '''
        ftype = kwargs.get("ftype", self.ftype)
        left_crop = kwargs.get('left_crop', self.left_crop)
        deformation_field = kwargs.get('deformation_field', self.deformation_field)
        perform_deformation = not np.all(np.isnan(deformation_field))
        if perform_deformation:
            perform_deformation_text = 'True'
        else:
            perform_deformation_text = 'False'

        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        I0 = kwargs.get('I0', self.Scaling[1,0])
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        TransformType = kwargs.get("TransformType", self.TransformType)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        Sample_ID = kwargs.get('Sample_ID', self.Sample_ID)
        SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
        Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        if BFMatcher:
            matcher = 'BFMatcher'
        else:
            matcher = 'FLANN'
        interpolation = kwargs.get('interpolation', self.interpolation)
        fill_value = kwargs.get('fill_value', 0)
        save_matches = kwargs.get("save_matches", self.save_matches)
        save_res_png  = kwargs.get("save_res_png", True )
        start = kwargs.get('start', 'edges')
        estimation = kwargs.get('estimation', 'interval')
        st = 1.0/np.sqrt(2.0)
        def_smoothing_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
        smoothing_kernel = kwargs.get('smoothing_kernel', def_smoothing_kernel)
        verbose = kwargs.get('verbose', True)
        dpi = kwargs.get('dpi', 600)
        int_results = pd.DataFrame()

        fl1 = self.fls.ravel()[index_pair[0]]
        fl2 = self.fls.ravel()[index_pair[1]]

        minmax = []
        for j,f in enumerate([fl1, fl2]):
            minmax.append(FIBSEM_frame(f, ftype=ftype, calculate_scaled_images=False).get_image_min_max(image_name = 'RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins))
        dmin = np.min(np.array(minmax))
        dmax = np.max(np.array(minmax))

        kpt_kwargs = {'ftype' : ftype,
                    'thr_min' : thr_min,
                    'thr_max' : thr_max,
                    'nbins' : nbins,
                    'SIFT_nfeatures' : SIFT_nfeatures,
                    'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
                    'SIFT_contrastThreshold' : SIFT_contrastThreshold,
                    'SIFT_edgeThreshold' : SIFT_edgeThreshold,
                    'SIFT_sigma' : SIFT_sigma,
                    'left_crop' : left_crop,
                    'use_existing_data' : False,
                    'save_deformed_image' : True,
                    'interpolation' : interpolation,
                    'fill_value' : fill_value,
                    'verbose' : verbose}
        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will perform SIFT evaluation using following parameters (kpt_kwargs):')
            print(kpt_kwargs)

        params_s3 = [[fl, dmin, dmax, kpt_kwargs] for fl in [fl1, fl2]]
        results_s3 = []
        for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ', display=verbose)):
            results_s3.append(extract_keypoints_descr_files(param_s3, deformation_field))
        fnms_kpts = [r[0] for r in results_s3]
        with open(fnms_kpts[0], 'rb') as f:
            kpp1s, des1, kpt_int1 = pickle.load(f)
        n_kpts1 = len(kpp1s)
        with open(fnms_kpts[1], 'rb') as f:
            kpp2s, des2, kpt_int2 = pickle.load(f)
        n_kpts2 = len(kpp2s)

        params_SIFT = []
        dt_kwargs = {'ftype' : ftype,
                'TransformType' : TransformType,
                'l2_matrix' : l2_matrix,
                'targ_vector': targ_vector, 
                'solver' : solver,
                'RANSAC_initial_fraction' : RANSAC_initial_fraction,
                'drmax' : drmax,
                'max_iter' : max_iter,
                'BFMatcher' : BFMatcher,
                'save_matches' : save_matches,
                'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
                'start' : start,
                'estimation' : estimation,
                'use_existing_data' : False,
                'verbose' : verbose}

        fname1 = fnms_kpts[0]
        fname2 = fnms_kpts[1]
        fnm_deformed1 = fname1.replace('_kpdes.bin','_def_image.tif')
        fnm_deformed2 = fname2.replace('_kpdes.bin','_def_image.tif')
        path_base, f1 = os.path.split(fname1)
        _, f2 = os.path.split(fname2)
        fnm_matches = os.path.join(path_base, f1.replace('_kpdes.bin', '_')+f2.replace('_kpdes.bin', '_matches.bin'))
        save_filename_default = os.path.join(self.data_dir, os.path.split(fnm_matches)[1].replace('_matches.bin', '_SIFT_test.png'))
        save_filename = kwargs.get('save_filename', save_filename_default)
        if verbose:
            print('Key-points files:')
            print(fname1)
            print(fname2)
            print('Deformed Image files:')
            print(fnm_deformed1)
            print(fnm_deformed2)
            print('Key-point matches file:')
            print(fnm_matches)
        dt_kwargs['fnm_matches'] = fnm_matches
        la = int(index_pair[0]) // self.n_tiles_per_layer
        ta = int(index_pair[0]) % self.n_tiles_per_layer
        lb = int(index_pair[1]) // self.n_tiles_per_layer
        tb = int(index_pair[1]) % self.n_tiles_per_layer
        dx = self.FirstPixels[lb, tb, 0] - self.FirstPixels[la, ta, 0]
        dy = self.FirstPixels[lb, tb, 1] - self.FirstPixels[la, ta, 1]
        dt_kwargs['warp_matrix'] = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        pair_index = np.where((self.index_pairs == index_pair).all(axis=1))[0]
        if len(pair_index) > 0:
            overlap_bounds = self.pair_overlap_bounds[pair_index[0]]
            dt_kwargs['overlap_bounds'] = overlap_bounds
        else:
            overlap_bounds = None
            dt_kwargs['image_margins'] = (self.YResolution, self.XResolution)   # full image fallback

        dt_kwargs['image_shape'] = (self.YResolution, self.XResolution)
        param_SIFT = [fname1, fname2, dt_kwargs]

        transformations_result = determine_transformations_files(param_SIFT)
        transform_matrix, fnm_matches, kpts, kpt_ints, error_abs_mean, error_FWHMx, error_FWHMy, iteration = transformations_result
        n_matches = len(kpts[0])

        if verbose:
            print('SIFT_transformation_matrix = ', transformations_result[0])
            print('SIFT_fnms_matches: ', transformations_result[1])
            print('SIFT_nmatches = ', n_matches)
            print('thr_min={:.0e}, thr_max={:.0e}'.format(thr_min, thr_max))
            print(TransformType.__name__+ ', ' + solver + ',  ' + matcher)
            print('SIFT_nfeatures={:d}'.format(SIFT_nfeatures))
            print('SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold))
            print('SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma))
            print('RANSAC_initial_fraction = {:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter))
            print('drmax={:.3f}'.format(drmax))
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   # of keypoints = {:d} and {:d}, # of matches = {:d}'.format(n_kpts1, n_kpts2, n_matches))
            if n_matches>0:
                src_pts_filtered, dst_pts_filtered = kpts
                src_intensities, dst_intensities = kpt_ints
                int_ratios = (dst_intensities-I0)/(src_intensities-I0)
                src_pts_transformed = src_pts_filtered @ transform_matrix[0:2, 0:2].T + transform_matrix[0:2, 2]
                xshifts = (dst_pts_filtered - src_pts_transformed)[:,0]
                yshifts = (dst_pts_filtered - src_pts_transformed)[:,1]
                print('Mean X-error={:.3f}, median X-error={:.3f}, FWHMx={:.3f}'.format(np.mean(xshifts), np.median(xshifts), error_FWHMx))
                print('Mean Y-error={:.3f}, median Y-error={:.3f}, FWHMy={:.3f}'.format(np.mean(yshifts), np.median(yshifts), error_FWHMy)) 
            else:
                print('No Matches detected')
            fs=12
            symsize = 2
            fsize_text = 5
            fsize_label = 10
            fsz = 10
            scale = 100
            width = 0.0010

            if n_matches>0:
                fig0, axs0 = plt.subplots(1,3, figsize=(10,3))
                fig0.subplots_adjust(left=0.06, bottom=0.18, right=0.99, top=0.87, wspace=0.20)
                axx = axs0[0]
                axx.set_xlabel('SIFT: X Error (pixels)')
                axx.set_ylabel('Count')
                axy = axs0[1]
                axy.set_xlabel('SIFT: Y Error (pixels)')
                axy.set_ylabel('Count')
                ax_int = axs0[2]
                ax_int.set_xlabel('Img1/Img0 Key-Pt Intensity Ratio')
                ax_int.set_ylabel('Count')
                axs0[0].text(0.05, 1.12, Sample_ID + ',  thr_min={:.0e}, thr_max={:.0e}, data range: {:.1f} ÷ {:.1f}, I0={:.1f}'.format(thr_min, thr_max, dmin, dmax, I0), transform=axs0[0].transAxes, fontsize=fsz)
                axs0[0].text(0.01, 1.03, 'SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}, SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold, SIFT_contrastThreshold, SIFT_sigma), fontsize=fsz, transform=axs0[0].transAxes)

                hist_int, bins_int, patches_int = ax_int.hist(int_ratios, bins = 64)
                FWHM_int, indi_int, inda_int, mx_int, mx_int_ind = find_FWHM(bins_int, hist_int[:-1], verbose=False, estimation=estimation, start=start, max_aver_aperture=5)
                if verbose:
                    print('Mean Int1/Int0 kpt ratio ={:.4f}, median Int1/Int0 kpt ratio={:.4f}, FWHMi={:.4f}'.format(np.mean(int_ratios), np.median(int_ratios), FWHM_int))
                db_int = (bins_int[1]-bins_int[0])/2.0
                ax_int.plot([bins_int[indi_int], bins_int[inda_int]], [mx_int/2.0, mx_int/2.0], 'r', linewidth = 4)
                ax_int.plot([bins_int[mx_int_ind]+db_int], [mx_int], 'rd')
                ax_int.text(0.05, 0.9, 'mean={:.4f}'.format(np.mean(int_ratios)), transform=ax_int.transAxes, fontsize=fsz)
                ax_int.text(0.05, 0.8, 'median={:.4f}'.format(np.median(int_ratios)), transform=ax_int.transAxes, fontsize=fsz)
                ax_int.text(0.05, 0.7, 'FWHM={:.4f}'.format(FWHM_int), transform=ax_int.transAxes, fontsize=fsz)

                xcounts, xbins, xhist_patches = axx.hist(xshifts, bins=64)
                error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, estimation=estimation, start=start, max_aver_aperture=5)
                dbx = (xbins[1]-xbins[0])/2.0
                axx.plot([xbins[indxi], xbins[indxa]], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
                axx.plot([xbins[mxx_ind]+dbx], [mxx], 'rd')
                axx.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(xshifts)), transform=axx.transAxes, fontsize=fsz)
                axx.text(0.05, 0.8, 'median={:.3f}'.format(np.median(xshifts)), transform=axx.transAxes, fontsize=fsz)
                axx.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMx), transform=axx.transAxes, fontsize=fsz)
                ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64)
                error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, estimation=estimation, start=start, max_aver_aperture=5)
                dby = (ybins[1]-ybins[0])/2.0
                axy.plot([ybins[indyi], ybins[indya]], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
                axy.plot([ybins[mxy_ind] + dby], [mxy], 'rd')
                axy.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(yshifts)), transform=axy.transAxes, fontsize=fsz)
                axy.text(0.05, 0.8, 'median={:.3f}'.format(np.median(yshifts)), transform=axy.transAxes, fontsize=fsz)
                axy.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMy), transform=axy.transAxes, fontsize=fsz)
                for ax in axs0.ravel():
                    ax.grid(True)
                if save_res_png:
                    save_filename0 = os.path.splitext(save_filename)[0] + '_plots.png'
                    axs0[0].text(0.0, -0.25, save_filename0, fontsize = 5, transform=axs0[0].transAxes)
                    fig0.savefig(save_filename0, dpi=dpi)

            fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))
            fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95, wspace=0.05)
            img1 = tiff.imread(fnm_deformed1)
            img2 = tiff.imread(fnm_deformed2)
            axs[0].imshow(img1, cmap='Greys')
            axs[1].imshow(img2, cmap='Greys')
            frame = FIBSEM_frame(fl1, read_header_only =True)
            axs[0].text(0.01, 1.00 - 0.015*frame.XResolution/frame.YResolution, 'thr_min={:.0e}, thr_max={:.0e}'.format(thr_min, thr_max), fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.035*frame.XResolution/frame.YResolution, TransformType.__name__+ ', ' + solver + ',  ' + matcher, fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.055*frame.XResolution/frame.YResolution, 'SIFT_nfeatures={:d}'.format(SIFT_nfeatures), fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.075*frame.XResolution/frame.YResolution, 'SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold), fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.095*frame.XResolution/frame.YResolution, 'SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma), fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.115*frame.XResolution/frame.YResolution, 'RANSAC_initial_fraction={:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter), fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.135*frame.XResolution/frame.YResolution, 'drmax={:.3f}'.format(drmax), fontsize=fsize_text, transform=axs[0].transAxes)
            axs[0].text(0.01, 1.00 - 0.155*frame.XResolution/frame.YResolution, 'Deformation Field Present: ' + perform_deformation_text + ',  left_crop={:d}'.format(left_crop), fontsize=fsize_text, transform=axs[0].transAxes)
            if overlap_bounds is not None:
                x_ov = int(overlap_bounds[1] - overlap_bounds[0])
                y_ov = int(overlap_bounds[3] - overlap_bounds[2])
                axs[0].text(0.01, 1.00 - 0.175*frame.XResolution/frame.YResolution, 'Overlap (x, y): {:d}, {:d}'.format(x_ov, y_ov), fontsize=fsize_text, transform=axs[0].transAxes)
            else:
                axs[0].text(0.01, 1.00 - 0.175*frame.XResolution/frame.YResolution, 'Overlap: full image', fontsize=fsize_text, transform=axs[0].transAxes)

            if n_matches > 0:
                columns_shifts=['X-src', 'Y-src', 'X-src transformed', 'Y-src transformed', 'X-dst', 'Y-dst', 'X-error', 'Y-error', 'Int-src', 'Int-dst']
                int_results = pd.DataFrame(np.vstack((np.array(src_pts_filtered).T, np.array(src_pts_transformed).T, np.array(dst_pts_filtered).T, xshifts, yshifts, src_intensities, dst_intensities)).T, columns = columns_shifts, index = None)

                x, y = src_pts_filtered.T
                M = np.sqrt(xshifts*xshifts+yshifts*yshifts)
                xs = xshifts
                ys = yshifts
                # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
                vec_field = axs[0].quiver(x,y,xs,ys,M, scale=scale, width = width, cmap='jet')
                cbar = fig.colorbar(vec_field, pad=0.05, shrink=0.70, orientation = 'horizontal', format="%.1f")
                cbar.set_label('SIFT Error Magnitude (pix)', fontsize=fsize_label)
                
                x, y = dst_pts_filtered.T
                # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
                vec_field = axs[1].quiver(x,y,xs,ys,M, scale=scale, width = width, cmap='jet')
                cbar = fig.colorbar(vec_field, pad=0.05, shrink=0.70, orientation = 'horizontal', format="%.1f")
                cbar.set_label('SIFT Error Magnitude (pix)', fontsize=fsize_label)

                axs[0].text(0.01, 1.00 - 0.195*frame.XResolution/frame.YResolution, '# of keypoints = {:d} and {:d}, # of matches ={:d}'.format(n_kpts1, n_kpts2, n_matches), fontsize=fsize_text, transform=axs[0].transAxes) 
                axs[0].text(0.01, 1.00 - 0.215*frame.XResolution/frame.YResolution, 'mean_error = {:.3f}, error_FWHMx = {:.3f},  error_FWHMy={:.3f}'.format(error_abs_mean, error_FWHMx, error_FWHMy), fontsize=fsize_text, transform=axs[0].transAxes) 

            for title, ax in zip([fnm_deformed1, fnm_deformed2], axs):
                ax.set_title(title, fontsize = fsize_text)
                ax.axis(False)

            if save_res_png:
                axs[0].text(0.0, -0.25, save_filename, fontsize = 5, transform=axs[0].transAxes)
                if verbose:
                    print('Summary Image is saved into file:')
                    print(save_filename)
                fig.savefig(save_filename, dpi=dpi)
                

        return fnm_deformed1, fnm_deformed2, transformations_result, int_results


    def determine_transformations_ECC(self, **kwargs):
        '''
        Determine transformation matrices for frame pairs using ECC. ©G.Shtengel 12/2025 gleb.shtengel@gmail.com
        Uses find_Transform_ECC(img1, img2, **kwargs).
        
        kwargs:
        ----------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        max_futures : int
            Max number of running DASK futures per batch. Default is self.max_futures (50000).
            Staged submission avoids overloading the scheduler with million-pair jobs.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        left_crop : int
            left image margin to be cropped off BEFORE distortion correction (via deformation field) is applied. Default is object attribute (or 0 if absent).
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is image attribute (or np.nan if absent - no distortion correction).
            Deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        motion : target transformation.
            Default is cv2.MOTION_TRANSLATION
        repeats : int
            repeat internally this many times. Default is 2.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
            Display intermediate results. Default is True.
        
        Returns:
        ----------
        transformations_results_3D : array of lists containing the results:
            [transformation_matrix, error_code]
            transformation_matrix : 2D np.float32 array
                Transformation matrix for each sequential frame pair.
            error_code : int
                CV2 error code.
        '''
        ftype = kwargs.get('ftype', self.ftype)
        repeats = kwargs.get('repeats', 2)
        verbose = kwargs.get('verbose', False)
        use_existing_data = kwargs.get('use_existing_data', False)
        left_crop = kwargs.get('left_crop', self.left_crop)
        deformation_field = kwargs.get('deformation_field', self.deformation_field)
        if hasattr(self, 'motion'):
            motion = kwargs.get('motion', self.motion)
        else:
            motion = kwargs.get('motion', cv2.MOTION_TRANSLATION)
        if hasattr(self, 'criteria'):
            criteria = kwargs.get('criteria', self.criteria)
        else:
            criteria = kwargs.get('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7))
        
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
        DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        use_existing_data = kwargs.get('use_existing_data', False)
        params_ECC = []
        fls = self.fls.ravel()

        for index_pair, pair_margins  in zip(tqdm(self.index_pairs, desc='Setting up ECC parameter list', display=verbose), self.pair_margins):
            dt_kwargs = {'ftype' : ftype,
                     'motion' : motion,
                     'criteria' : criteria,
                     'use_existing_data' : use_existing_data,
                     'verbose' : verbose,
                     'left_crop' : left_crop,
                     'image_margins' : pair_margins}
            fname1 = fls[index_pair[0]]
            fname2 = fls[index_pair[1]]
            la = int(index_pair[0]) // self.n_tiles_per_layer
            ta = int(index_pair[0]) % self.n_tiles_per_layer
            lb = int(index_pair[1]) // self.n_tiles_per_layer
            tb = int(index_pair[1]) % self.n_tiles_per_layer
            dx = self.FirstPixels[lb, tb, 0] - self.FirstPixels[la, ta, 0]
            dy = self.FirstPixels[lb, tb, 1] - self.FirstPixels[la, ta, 1]
            dt_kwargs['warp_matrix'] = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
            param_ECC = [fname1, fname2, dt_kwargs]
            params_ECC.append(param_ECC)

        max_futures = kwargs.get('max_futures', self.max_futures)
        if use_DASK:
            # Scatter deformation_field once — all tasks reference it via the future.
            shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
            # Stage submissions to avoid choking the scheduler on million-pair jobs.
            transformations_results_3D = []
            n_tasks   = len(params_ECC)
            n_batches = (n_tasks + max_futures - 1) // max_futures
            for DASK_batch in tqdm(range(n_batches), desc='determine_transformations_ECC DASK batches'):
                start = DASK_batch * max_futures
                stop  = min(start + max_futures, n_tasks)
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Starting DASK batch {:d}/{:d} with {:d} jobs ({:d} remaining after this batch)'.format(
                                DASK_batch + 1, n_batches, stop - start, n_tasks - stop))
                futures_ECC = DASK_client.map(find_Transform_ECC_DASK,
                                              params_ECC[start:stop],
                                              deformation_field=shared_data_future,
                                              retries=DASK_client_retries)
                transformations_results_3D += DASK_client.gather(futures_ECC)
        else:
            transformations_results_3D = []
            for param_ECC in tqdm(params_ECC, desc = 'Extracting transformation parameters: ', display=verbose):
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Determining transformation params for:')
                    print(param_ECC)
                transformations_results_3D.append(find_Transform_ECC_DASK(param_ECC, deformation_field))
        
        for j, transformations_result  in enumerate(tqdm(transformations_results_3D, desc = 'Parsing the ECC results', display = verbose)):
            try:
                self.ECC_transformation_matrices[j, 0:2, :] = np.nan_to_num(transformations_result[0])
                self.ECC_transformation_valid[j] = transformations_result[1] == 0
            except Exception as e:
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   An error occurred: {}'.format(e))
                    print('transformations_result:  ', transformations_result)
        return transformations_results_3D


    def solve_stack_stitching(self, **kwargs):
        '''
        Solve mosaic stack stitching (perform bundle optimization). ©G.Shtengel 01/2026 gleb.shtengel@gmail.com

        kwargs:
        ----------
        initialize_transformation_first : bool
            if True (default), re-initialize the tr_matr first.
        verbose : boolean
            Display intermediate results. Default is False.
        method : string
            Options are: ['SIFT-ECC', 'SIFT', 'ECC', 'SIFT-Affine']. Default is 'ECC'.
            'SIFT-ECC' means - try SIFT first, and for the tiles that SIFT failed, try ECC.
            'SIFT-Affine' performs a keypoint-based bundle adjustment that solves for a full
            per-tile affine transform (scale, rotation, shear, and translation) using all
            inlier SIFT keypoint matches.  Requires that determine_transformations_SIFT has
            already been run with save_matches=True so that self.SIFT_fnms_matches is
            populated.  The residual 2-norm is stored in self.SIFT_Affine_r2norm.
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-. Default is [True, True].
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
        subtract_FOVtrend_from_fit : [boolean, boolean]
            If True, FOV trends (image shifts performed during imaging) will be subtracted first, so they do not bias the linear trends in the line above.
            Default is [True, True].
        Returns:
        ----------
        tile_positions : ndarray, shape (nz_tiles, n_tiles_per_layer, 2)
            Tile positions derived on the fly from tr_matr as -tr_matr[:, :, 0:2, 2].
            Only tiles with at least one valid constraint are updated;
            all other tiles retain their initialised tr_matr values.
        '''
        initialize_transformation_first = kwargs.get('initialize_transformation_first', True)
        verbose = kwargs.get('verbose', False)
        method = kwargs.get('method', 'ECC')
        valid_methods = ['SIFT-ECC', 'SIFT', 'ECC', 'SIFT-Affine']
        subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
        subtract_FOVtrend_from_fit = kwargs.get("subtract_FOVtrend_from_fit", [True, True])

        w_sqrt_intra = np.sqrt(self.intralayer_weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||
        w_sqrt_inter = np.sqrt(self.interlayer_weight)
        weights = np.concatenate((np.full((self.nh+self.nv), w_sqrt_intra), np.full(self.nl, w_sqrt_inter)))
        if initialize_transformation_first:
            self.tr_matr = self.default_tr_matr.copy()

        if method not in valid_methods:
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Method ' + method +' is not among valid methods: ', valid_methods)
            return np.nan

        if method == 'SIFT-Affine':
            # ------------------------------------------------------------------
            # Keypoint-based bundle adjustment — full affine transform per tile.
            # All per-tile tr_matr updates are handled inside _solve_affine_bundle;
            # subtract_linear_fit post-processing and the return below are shared
            # with the ShiftTransform paths.
            # ------------------------------------------------------------------
            valid_tile_flat = self._solve_affine_bundle(**kwargs)

        else:
            # ------------------------------------------------------------------
            # ShiftTransform paths: 'SIFT', 'ECC', 'SIFT-ECC'
            # ------------------------------------------------------------------
            if method == 'SIFT':
                self.SIFT_residual_error_x = np.full(self.C, np.nan)
                self.SIFT_residual_error_y = np.full(self.C, np.nan)
                bx = self.SIFT_transformation_matrices[:, 0, 2] * weights
                by = self.SIFT_transformation_matrices[:, 1, 2] * weights
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + f',  Solving for X displacement using method {method}')
                res_x_all = lsqr(self.A_csr[self.SIFT_transformation_valid], bx[self.SIFT_transformation_valid])
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + f',  Solving for Y displacement using method {method}')
                res_y_all = lsqr(self.A_csr[self.SIFT_transformation_valid], by[self.SIFT_transformation_valid])
                # calculate weighted residuals: b_weighted - A_weighted x
                self.SIFT_residual_error_x[self.SIFT_transformation_valid] = bx[self.SIFT_transformation_valid] - self.A_csr[self.SIFT_transformation_valid] @ res_x_all[0]
                self.SIFT_residual_error_y[self.SIFT_transformation_valid] = by[self.SIFT_transformation_valid] - self.A_csr[self.SIFT_transformation_valid] @ res_y_all[0]
                self.SIFT_r2norm_x = res_x_all[4]
                self.SIFT_r2norm_y = res_y_all[4]
                # Valid-constraint mask for this method — used below to identify
                # which tiles should have their tr_matr updated.
                valid_constraint_mask = self.SIFT_transformation_valid
            else:
                self.ECC_residual_error_x = np.full(self.C, np.nan)
                self.ECC_residual_error_y = np.full(self.C, np.nan)
                bx = self.ECC_transformation_matrices[:, 0, 2] * weights
                by = self.ECC_transformation_matrices[:, 1, 2] * weights
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + f',  Solving for X displacement using method {method}')
                res_x_all = lsqr(self.A_csr[self.ECC_transformation_valid], bx[self.ECC_transformation_valid])
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') + f',  Solving for Y displacement using method {method}')
                res_y_all = lsqr(self.A_csr[self.ECC_transformation_valid], by[self.ECC_transformation_valid])
                self.ECC_residual_error_x[self.ECC_transformation_valid] = bx[self.ECC_transformation_valid] - self.A_csr[self.ECC_transformation_valid] @ res_x_all[0]
                self.ECC_residual_error_y[self.ECC_transformation_valid] = by[self.ECC_transformation_valid] - self.A_csr[self.ECC_transformation_valid] @ res_y_all[0]
                self.ECC_r2norm_x = res_x_all[4]
                self.ECC_r2norm_y = res_y_all[4]
                # Valid-constraint mask for this method — used below to identify
                # which tiles should have their tr_matr updated.
                valid_constraint_mask = self.ECC_transformation_valid

            res_x = res_x_all[0]
            res_y = res_y_all[0]
            positions = np.zeros((self.nz_tiles * self.n_tiles_per_layer, 2))
            positions[:, 0] = res_x
            positions[:, 1] = res_y
            positions_3d = positions.reshape((self.nz_tiles, self.n_tiles_per_layer, 2))

            # Determine which tiles appear in at least one valid pairwise constraint.
            # self.index_pairs has shape (C, 2): each row holds the two flat tile
            # indices for that constraint row in A_csr.  Collecting the unique indices
            # from all valid rows gives exactly the tiles whose lsqr-solved positions
            # are meaningful (i.e. anchored by real image data).
            valid_tile_flat = np.unique(self.index_pairs[valid_constraint_mask])  # flat 1D tile indices
            valid_z = valid_tile_flat // self.n_tiles_per_layer   # layer index
            valid_t = valid_tile_flat  % self.n_tiles_per_layer   # within-layer tile index

            # Update tr_matr only for tiles that had at least one valid constraint.
            # Tiles with no valid constraints keep their existing tr_matr translations
            # (i.e. the nominal/initialised positions), since the solver has no real
            # data to constrain their positions and would otherwise write arbitrary values.
            # dx and dy are average shifts of new positions relative to default positions.
            dx = np.mean(self.tr_matr[valid_z, valid_t, 0, 2] - positions_3d[valid_z, valid_t, 0])
            dy = np.mean(self.tr_matr[valid_z, valid_t, 1, 2] - positions_3d[valid_z, valid_t, 1])

            self.tr_matr[valid_z, valid_t, 0, 2] = positions_3d[valid_z, valid_t, 0] + dx
            self.tr_matr[valid_z, valid_t, 1, 2] = positions_3d[valid_z, valid_t, 1] + dy

        # ------------------------------------------------------------------
        # Post-processing shared by ALL methods: subtract linear drift from
        # the translation columns, then print a summary if verbose.
        # ------------------------------------------------------------------
        if subtract_linear_fit[0]:
            Xshift_mean = np.mean((self.tr_matr[:, :, 0, 2] - self.tr_matr[0, :, 0, 2]), axis=1)
            fr = np.arange(0, len(Xshift_mean))
            pX = np.polyfit(fr, Xshift_mean, 1)
            Xfit = np.polyval(pX, fr)
            self.tr_matr[:, :, 0, 2] -= Xfit[:, np.newaxis]

        if subtract_linear_fit[1]:
            Yshift_mean = np.mean((self.tr_matr[:, :, 1, 2] - self.tr_matr[0, :, 1, 2]), axis=1)
            fr = np.arange(0, len(Yshift_mean))
            pY = np.polyfit(fr, Yshift_mean, 1)
            Yfit = np.polyval(pY, fr)
            self.tr_matr[:, :, 1, 2] -= Yfit[:, np.newaxis]

        if verbose:
            n_total = self.nz_tiles * self.n_tiles_per_layer
            n_updated = len(valid_tile_flat)
            n_skipped = n_total - n_updated
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + f':   solve_stack_stitching ({method}): '
                  f'updated {n_updated}/{n_total} tiles; '
                  f'{n_skipped} tile(s) with no valid constraints left unchanged.')

        # Return tile positions derived on the fly from tr_matr.
        # tr_matr[:,:,i,2] stores the negative translation, so negate to get
        # positive (x, y) pixel positions in canvas space.
        # tile_positions = -self.tr_matr[:, :, 0:2, 2]
        return -self.tr_matr[:, :, 0:2, 2]

    def _solve_affine_bundle(self, **kwargs):
        '''
        Keypoint-based bundle adjustment that solves for a full affine transform
        per tile.  Called internally by solve_stack_stitching when method='SIFT-Affine'.
        ©G.Shtengel 01/2026 gleb.shtengel@gmail.com

        Parameterisation (Option-B / minimum-norm anchoring)
        -----------------------------------------------------
        Each tile k carries 6 unknowns representing the *deviation* from the
        identity transform:
            φ_k = [da_k, db_k, dtx_k,  dc_k, dd_k, dty_k]
        so that the full tile-to-canvas affine is:
            M_k = I + δ_k  =  [[1+da_k,    db_k,  dtx_k],
                                [  dc_k,  1+dd_k,  dty_k],
                                [     0,       0,      1]]

        For every inlier match  p = [px, py]  in tile_i  ↔  q = [qx, qy]  in tile_j
        the canvas-consistency condition  M_i @ p̃ = M_j @ q̃  expands to two linear
        equations with a *non-zero* RHS:
            x:  da_i·px + db_i·py + dtx_i − da_j·qx − db_j·qy − dtx_j  =  qx − px
            y:  dc_i·px + dd_i·py + dty_i − dc_j·qx − dd_j·qy − dty_j  =  qy − py

        This yields a sparse system  B φ = r  of shape  (2·M_total, 6·V)  which is
        solved in one lsqr call.  lsqr's minimum-norm solution distributes deviations
        from identity evenly across all tiles; no explicit anchor tile is needed.

        After the solve the per-tile dtx / dty values are centred onto the nominally-
        initialised tr_matr positions via a global dx/dy correction (the same approach
        used by the ShiftTransform path).  The non-translation parameters (da, db, dc,
        dd) are written directly — their minimum-norm values are already centred near 0.

        Intra-layer pairs are weighted by sqrt(intralayer_weight) and inter-layer pairs
        by sqrt(interlayer_weight), mirroring the ShiftTransform path.

        Parameters
        ----------
        **kwargs forwarded from solve_stack_stitching:
            verbose : bool, default False

        Returns
        -------
        valid_tile_flat : 1-D int ndarray
            Flat tile indices (k = layer * n_tiles_per_layer + tile_in_layer) that
            appear in at least one valid match pair.  Used by the caller for the
            verbose tile-count summary.

        Side effects
        ------------
        self.tr_matr[valid_z, valid_t]   updated with solved affine matrices
        self.SIFT_Affine_r2norm          set to the lsqr residual 2-norm
        '''
        verbose  = kwargs.get('verbose', False)
        n        = self.n_tiles_per_layer
        V        = self.nz_tiles * n
        w_sqrt_intra = np.sqrt(self.intralayer_weight)
        w_sqrt_inter = np.sqrt(self.interlayer_weight)

        # ------------------------------------------------------------------ #
        # 1.  Identify valid pairs and pre-count total matches                #
        # ------------------------------------------------------------------ #
        valid_pair_indices = np.where(self.SIFT_transformation_valid)[0]
        if len(valid_pair_indices) == 0:
            if verbose:
                print('  _solve_affine_bundle: no valid SIFT matches found.')
            return np.array([], dtype=int)

        M_total = int(self.SIFT_nmatches[valid_pair_indices].sum())
        n_rows  = 2 * M_total    # two equations (x and y) per match
        n_cols  = 6 * V          # six unknowns per tile

        # ------------------------------------------------------------------ #
        # 2.  Build sparse COO arrays (vectorised per pair, no per-match loop)#
        #                                                                      #
        # Each match contributes 12 non-zero entries total:                   #
        #   tile_i: 3 entries in the x-row + 3 entries in the y-row = 6      #
        #   tile_j: 3 entries in the x-row + 3 entries in the y-row = 6      #
        # ------------------------------------------------------------------ #
        nnz_max = 12 * M_total
        B_row  = np.empty(nnz_max, dtype=np.int64)
        B_col  = np.empty(nnz_max, dtype=np.int64)
        B_data = np.empty(nnz_max, dtype=np.float64)
        rhs    = np.zeros(n_rows,  dtype=np.float64)

        ptr  = 0    # write pointer into B_row / B_col / B_data
        mptr = 0    # next free row index in the equation system

        for j in valid_pair_indices:
            fnm = self.SIFT_fnms_matches[j]
            try:
                with open(fnm, 'rb') as fh:
                    match_data = pickle.load(fh)
            except Exception:
                if verbose:
                    print(f'  _solve_affine_bundle: could not load match file for pair {j}: {fnm}')
                continue

            src_pts, dst_pts = match_data[1]   # each (M, 2), full tile-local coords
            M = len(src_pts)
            if M == 0:
                continue

            px = src_pts[:, 0].astype(np.float64)   # (M,)
            py = src_pts[:, 1].astype(np.float64)
            qx = dst_pts[:, 0].astype(np.float64)
            qy = dst_pts[:, 1].astype(np.float64)

            tile_i, tile_j = self.index_pairs[j]
            is_inter = (int(tile_i) // n) != (int(tile_j) // n)
            w  = w_sqrt_inter if is_inter else w_sqrt_intra

            x_rows = mptr + np.arange(M, dtype=np.int64) * 2   # row indices for x-equations
            y_rows = x_rows + 1                                  # row indices for y-equations
            ones   = np.ones(M, dtype=np.float64)
            ci     = int(tile_i) * 6    # column base for tile_i  (φ_i starts at col ci)
            cj     = int(tile_j) * 6    # column base for tile_j

            # -- tile_i x-row: columns ci+0,ci+1,ci+2  ←  da_i·px + db_i·py + dtx_i
            for k_off, vals in enumerate([px, py, ones]):
                B_row [ptr:ptr+M] = x_rows
                B_col [ptr:ptr+M] = ci + k_off
                B_data[ptr:ptr+M] = w * vals
                ptr += M

            # -- tile_i y-row: columns ci+3,ci+4,ci+5  ←  dc_i·px + dd_i·py + dty_i
            for k_off, vals in enumerate([px, py, ones]):
                B_row [ptr:ptr+M] = y_rows
                B_col [ptr:ptr+M] = ci + 3 + k_off
                B_data[ptr:ptr+M] = w * vals
                ptr += M

            # -- tile_j x-row: columns cj+0,cj+1,cj+2  ←  −(da_j·qx + db_j·qy + dtx_j)
            for k_off, vals in enumerate([qx, qy, ones]):
                B_row [ptr:ptr+M] = x_rows
                B_col [ptr:ptr+M] = cj + k_off
                B_data[ptr:ptr+M] = -w * vals
                ptr += M

            # -- tile_j y-row: columns cj+3,cj+4,cj+5  ←  −(dc_j·qx + dd_j·qy + dty_j)
            for k_off, vals in enumerate([qx, qy, ones]):
                B_row [ptr:ptr+M] = y_rows
                B_col [ptr:ptr+M] = cj + 3 + k_off
                B_data[ptr:ptr+M] = -w * vals
                ptr += M

            # -- RHS: deviation from identity means measured relative displacement
            rhs[x_rows] = w * (qx - px)
            rhs[y_rows] = w * (qy - py)

            mptr += 2 * M

        # Trim pre-allocated arrays to the number of entries actually written
        # (some pairs might have been skipped due to missing files or M==0).
        B_row  = B_row [:ptr]
        B_col  = B_col [:ptr]
        B_data = B_data[:ptr]
        n_rows_actual = mptr

        if n_rows_actual == 0:
            if verbose:
                print('  _solve_affine_bundle: no match equations could be built.')
            return np.array([], dtype=int)

        # ------------------------------------------------------------------ #
        # 3.  Solve  B φ = r  (minimum-norm least-squares)                    #
        # ------------------------------------------------------------------ #
        B_csr = csr_matrix((B_data, (B_row, B_col)), shape=(n_rows_actual, n_cols))
        res   = lsqr(B_csr, rhs[:n_rows_actual])
        phi   = res[0]                     # shape (6·V,)
        self.SIFT_Affine_r2norm = res[4]   # weighted residual 2-norm

        if verbose:
            print(f'  _solve_affine_bundle: solved {n_rows_actual} equations for {n_cols} unknowns; '
                  f'r2norm = {self.SIFT_Affine_r2norm:.4g}')

        # ------------------------------------------------------------------ #
        # 4.  Reconstruct tr_matr from φ                                      #
        #                                                                      #
        # φ_k = [da_k, db_k, dtx_k,  dc_k, dd_k, dty_k]                      #
        # M_k = [[1+da_k,   db_k,  dtx_k],                                    #
        #        [  dc_k, 1+dd_k,  dty_k],                                    #
        #        [     0,      0,      1]]                                     #
        # ------------------------------------------------------------------ #
        phi_3d = phi.reshape(V, 6)   # (V, 6) — one row per flat tile index

        # Determine which tiles appeared in at least one valid match pair.
        valid_tile_flat = np.unique(self.index_pairs[self.SIFT_transformation_valid])
        valid_z = valid_tile_flat // n
        valid_t = valid_tile_flat  % n

        # Extract solved deviations for the constrained tiles (vectorised).
        da_arr  = phi_3d[valid_tile_flat, 0]
        db_arr  = phi_3d[valid_tile_flat, 1]
        dtx_arr = phi_3d[valid_tile_flat, 2]
        dc_arr  = phi_3d[valid_tile_flat, 3]
        dd_arr  = phi_3d[valid_tile_flat, 4]
        dty_arr = phi_3d[valid_tile_flat, 5]

        # Centre the solved translations onto the nominal (initialised) tr_matr
        # positions — exactly as done in the ShiftTransform path (dx/dy correction).
        # The minimum-norm solve yields dtx/dty values close to 0; adding dx/dy
        # shifts them to match the expected canvas positions.
        dx = np.mean(self.tr_matr[valid_z, valid_t, 0, 2] - dtx_arr)
        dy = np.mean(self.tr_matr[valid_z, valid_t, 1, 2] - dty_arr)

        # Write full affine matrices for all constrained tiles (vectorised).
        self.tr_matr[valid_z, valid_t, 0, 0] = 1.0 + da_arr
        self.tr_matr[valid_z, valid_t, 0, 1] = db_arr
        self.tr_matr[valid_z, valid_t, 0, 2] = dtx_arr + dx
        self.tr_matr[valid_z, valid_t, 1, 0] = dc_arr
        self.tr_matr[valid_z, valid_t, 1, 1] = 1.0 + dd_arr
        self.tr_matr[valid_z, valid_t, 1, 2] = dty_arr + dy
        # Row 2 is [0, 0, 1] — already set by initialize_transformation_first.

        return valid_tile_flat

    def solve_intensity_normalization(self, **kwargs):
        '''
        Solve for per-tile multiplicative intensity scale factors using bundle optimization.
        ©G.Shtengel gleb.shtengel@gmail.com

        For each valid adjacent tile pair the ratio of mean matched keypoint intensities
        (after dark-count subtraction) constrains the relative scale between the two tiles.
        The same sparse constraint matrix (A_csr) used for position bundle optimization is
        reused. lsqr minimizes the total weighted log-scale residual across all pairs,
        yielding one scale factor per tile. The result is normalized so that the mean
        log-scale is zero (i.e. the geometric mean of all scale factors is 1).

        kwargs:
        ----------
        method : str
          Source of intensity ratios. Options:
            'SIFT'               — matched keypoint intensities (whole-tile).
                                   Requires prior determine_transformations_SIFT().
            'mean'               — whole-frame mean per tile.
                                   Requires prior compute_frame_intensity_ratios(method='mean').
            'percentile'         — whole-frame percentile per tile.
                                   Requires prior compute_frame_intensity_ratios(method='percentile').
            'overlap_mean'       — mean over each pair's overlap ROI.
                                   Requires prior compute_overlap_intensity_ratios(method='mean', DASK_client=...).
            'overlap_percentile' — percentile over each pair's overlap ROI.
                                   Requires prior compute_overlap_intensity_ratios(method='percentile', DASK_client=...).
          Default is 'SIFT'.
        I0 : float
          Dark-count offset subtracted before computing ratios. Default is self.Scaling[1, 0].
        intralayer_weight : float
          Weight for intra-layer pair constraints. Default is self.intralayer_weight.
        interlayer_weight : float
          Weight for inter-layer pair constraints. Default is self.interlayer_weight.
        tikhonov_damp : float
          Strength of L2 regularization on log-scales (Tikhonov). Each tile's log-scale
          is pulled toward 0 (i.e., scale → 1.0) with strength `tikhonov_damp`.
          Reduces drift of poorly-connected tiles (corners/edges) at the cost of
          slightly under-correcting well-connected tiles.
          Reasonable range - 0 to 30 in log-scale:
            0.0 disables regularization (default).
              ~1  gentle anchor; barely touches strong tiles, mildly nudges corners
              ~3 – 10 meaningful anchor on corners; some pullback on strong tiles too
              30+ starts to dominate — all scales pushed toward 1.0 regardless of data
        target_damp : float
          Strength of regularization pulling each pair's solved log-ratio
          (log_scale[b] - log_scale[a]) toward a per-pair target log(T_ab),
          where T_ab is determined by the detectors that recorded tiles a and b
          (see compute_detector_target_intensity_ratios). Encodes the prior that pair
          differences are explained largely by known detector sensitivity
          differences. 0.0 disables this term (default).
          Coexists with `tikhonov_damp` (both can be nonzero).

        verbose : boolean
          Display intermediate results. Default is False.

        Returns:
        ----------
        tile_scales : 2D np.float32 array, shape (nz_tiles, n_tiles_per_layer)
          Multiplicative scale factor to apply to each tile before assembling the mosaic.
          Stored as self.tile_scales.
        '''
        verbose = kwargs.get('verbose', False)
        method = kwargs.get('method', 'SIFT')
        I0 = kwargs.get('I0', self.Scaling[1, 0])
        intralayer_weight = kwargs.get('intralayer_weight', self.intralayer_weight)
        interlayer_weight = kwargs.get('interlayer_weight', self.interlayer_weight)
        tikhonov_damp = kwargs.get('tikhonov_damp', 0.0)
        target_damp = kwargs.get('target_damp', 0.0)

        w_sqrt_intra = np.sqrt(intralayer_weight)
        w_sqrt_inter = np.sqrt(interlayer_weight)
        weights = np.concatenate((np.full(self.nh + self.nv, w_sqrt_intra),
                                np.full(self.nl, w_sqrt_inter)))
        if method == 'SIFT':
            ratios = self.SIFT_intensity_ratios
            valid  = self.SIFT_transformation_valid & np.isfinite(ratios) & (ratios > 0)
        elif method in ('mean', 'percentile'):
            attr = method + '_intensity_ratios'
            ratios = getattr(self, attr)
            if not np.any(np.isfinite(ratios)):
                raise RuntimeError(
                    "{} has not been computed yet. Call "
                    "compute_frame_intensity_ratios(method='{}') first "
                    "(requires prior evaluate_FIBSEM_statistics(); the percentile is "
                    "set there as self.percentile, not at this call).".format(attr, method))
            valid  = np.isfinite(ratios) & (ratios > 0)
        elif method in ('overlap_mean', 'overlap_percentile'):
            base = method[len('overlap_'):]   # 'mean' or 'percentile'
            attr = 'overlap_' + base + '_intensity_ratios'
            ratios = getattr(self, attr)
            if not np.any(np.isfinite(ratios)):
                raise RuntimeError(
                    "{} has not been computed yet. Call "
                    "compute_overlap_intensity_ratios(method='{}', DASK_client=...) "
                    "first.".format(attr, base))
            valid  = self.overlap_intensity_ratios_valid & np.isfinite(ratios) & (ratios > 0)
        else:
            raise ValueError(
                "method '{}' not supported. Use 'SIFT', 'mean', 'percentile', "
                "'overlap_mean', or 'overlap_percentile'.".format(method))

        if np.sum(valid) == 0:
          if verbose:
              print('No valid intensity ratios found. Returning unit scales.')
          self.tile_scales = np.ones((self.nz_tiles, self.n_tiles_per_layer))
          return self.tile_scales

        log_ratios = np.log(ratios[valid]) * weights[valid]

        if target_damp > 0.0:
            if not np.any(self.target_intensity_ratios_valid):
                raise RuntimeError(
                    "target_intensity_ratios not available. Call compute_detector_target_intensity_ratios() first.")
            from scipy.sparse import csr_matrix, vstack
            t_valid = self.target_intensity_ratios_valid & np.isfinite(self.target_intensity_ratios) & (self.target_intensity_ratios > 0)
            if verbose:
                print('Detector prior: stacking {} target rows (target_damp={:.4g})'.format(
                    int(t_valid.sum()), target_damp))
            # Build A_reg from scratch with uniform ±target_damp entries (one row per valid
            # pair). Independent of A_csr's pre-weighted row scaling, so this stays correct
            # when the caller overrides intralayer_weight / interlayer_weight kwargs.
            t_indices = np.where(t_valid)[0]
            n_reg     = len(t_indices)
            pairs     = self.index_pairs[t_indices]                # (n_reg, 2)
            rows      = np.repeat(np.arange(n_reg), 2)
            cols      = pairs.ravel().astype(np.int64)
            data      = np.tile([-target_damp, target_damp], n_reg).astype(np.float64)
            A_reg     = csr_matrix((data, (rows, cols)), shape=(n_reg, self.A_csr.shape[1]))
            b_reg     = target_damp * np.log(self.target_intensity_ratios[t_valid])
            A_aug = vstack([self.A_csr[valid], A_reg], format='csr')
            b_aug = np.concatenate([log_ratios, b_reg])
            res = lsqr(A_aug, b_aug, damp=tikhonov_damp)
        else:
            res = lsqr(self.A_csr[valid], log_ratios, damp=tikhonov_damp)
        log_scales = res[0]

        # Normalise: geometric mean of all scale factors = 1
        log_scales -= np.mean(log_scales)
        tile_scales = 1.0 / np.exp(log_scales).reshape(self.nz_tiles, self.n_tiles_per_layer)

        if verbose:
            print('Intensity normalization method: ' + method)
            print('Intensity normalization parameters (tikhonov_damp={:.4g}, target_damp={:.4g}): '
                'intralayer_weight={:.4f}, interlayer_weight={:.4f}'.format( tikhonov_damp, target_damp, intralayer_weight, interlayer_weight))
            print('Intensity scale factors: min={:.4f}, max={:.4f}, std={:.4f}'.format(np.min(tile_scales), np.max(tile_scales), np.std(tile_scales)))

        self.tile_scales = tile_scales
        return tile_scales


    def generate_transformation_report(self, **kwargs):
        '''
        Generate Report Plot for transformation summary. ©G.Shtengel 12/2022 gleb.shtengel@gmail.com

        kwargs:
        ----------
        frame_inds : array or list
            Array of frame/layer indices to plot; default is np.arange(self.nz_tiles).
        Sample_ID : str
            Sample label for the plot title; default is self.Sample_ID.
        n_tiles_per_layer : int
            Number of tiles to iterate over; default is self.n_tiles_per_layer.
        data_dir : path
            Directory used as fallback for save_fname; default is self.data_dir.
        tile_id : int
            tile ID to show. Default is 0.
        save_png : boolean
            If True (default), the plot is saved into PNG file.
        dpi : int
            DPI for PNG. Default is 300.
        save_fname : string
            File name to save the PNG image. Default is os.path.join(data_dir, 'Relative_Tile_Shifts.png').
        verbose : boolean
            Display intermediate results. Default is False.

        Returns:
        ----------
        save_fname

        '''
        n_tiles_per_layer = kwargs.get('n_tiles_per_layer', self.n_tiles_per_layer)
        tile_id = kwargs.get('tile_id', 0)
        verbose = kwargs.get('verbose', False)
        save_png = kwargs.get('save_png', True)
        dpi = kwargs.get('dpi', 300)
        data_dir = kwargs.get('data_dir', self.data_dir)
        if save_png:
            try:
                save_fname = kwargs.get('save_fname', os.path.splitext(self.fnm_mosaic_stack)[0] + '_Relative_Tile_Shifts.png')
            except:
                save_fname = kwargs.get('save_fname', os.path.join(data_dir, 'Relative_Tile_Shifts.png'))
        else:
            save_fname = 'Image not saved'
        Sample_ID = kwargs.get('Sample_ID', self.Sample_ID)
        frame_inds = kwargs.get('frame_inds', np.arange(self.nz_tiles))
        # Derive relative tile positions from tr_matr (tr_matr[:,:,i,2] = -position_i).
        # Subtracting frame 0 gives positions relative to the first layer.
        tile_positions_x = - self.tr_matr[frame_inds, :, 0, 2]
        tile_positions_y = - self.tr_matr[frame_inds, :, 1, 2]

        if verbose:
            print('Generating Plot')
        fig, axs = plt.subplots(3,1, figsize = (6,10), sharex=True)
        fig.subplots_adjust(left=0.12, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.03)

        for k in np.arange(n_tiles_per_layer):
            my_col = plt.get_cmap("gist_rainbow_r")((n_tiles_per_layer-k)/(n_tiles_per_layer-1))
            tile_positions_xk = tile_positions_x[:, k] - np.mean(tile_positions_x[:, k])
            tile_positions_yk = tile_positions_y[:, k] - np.mean(tile_positions_y[:, k])
            if k == tile_id:
                axs[0].plot(frame_inds, tile_positions_xk, color=my_col, marker='x', markersize=4, label='Tile {:d}, X-shift'.format(tile_id))
                axs[1].plot(frame_inds, tile_positions_yk, color=my_col, marker='x', markersize=4, label='Tile {:d}, Y-shift'.format(tile_id))
                axs[2].plot(frame_inds, tile_positions_xk, color='red', linewidth = 0.25, label='Tile {:d}, X-shift'.format(tile_id))
                axs[2].plot(frame_inds, tile_positions_yk, color='blue', linewidth = 0.25, label='Tile {:d}, Y-shift'.format(tile_id))
            else:
                axs[0].plot(frame_inds, tile_positions_xk, color=my_col, linewidth = 0.25)
                axs[1].plot(frame_inds, tile_positions_yk, color=my_col, linewidth = 0.25)

        for ax in axs:
            ax.grid(True)
            ax.legend(fontsize=12, loc='lower right')

        axs[0].text(0.40, 0.92, 'All Tiles: X-shift', transform=axs[0].transAxes, fontsize=12)
        axs[0].text(0.2, 1.03, Sample_ID, transform=axs[0].transAxes, fontsize=12)
        axs[1].text(0.40, 0.92, 'All Tiles: Y-shift', transform=axs[1].transAxes, fontsize=12)
        axs[2].set_xlabel('Frame')
        axs[0].set_ylabel('Relative X-Shift (pix)')
        axs[1].set_ylabel('Relative Y-Shift (pix)')
        axs[2].set_ylabel('Relative Shift (pix)')
        if save_png:
            axs[2].text(-0.1, -0.18, save_fname, transform=axs[2].transAxes, fontsize=5)
            fig.savefig(save_fname, dpi=dpi)
        return save_fname


    def assemble_layer_mosaic(self, layer_id, **kwargs):
        '''
        Assemble layer mosaic based on transformation matrices for each tile. Options to save snapshot, save mosaic as FIBSEM_frame (dat file) or save_images as JPG or PNG. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com

        Parameters:
        ----------
        layer_id : int
            Layer ID should be a value between -1 and self.nz_tiles-1. -1 means the last layer will be assembled.
        
        kwargs:
        ----------
        image_names : str
            String of Image names. Default is ['RawImageA', 'RawImageB'] if both are available, ['RawImageA'] otherwise.
        weight_min : float
            vmin for weight. Default is 1.
        weight_max : float
            vmax for weight. Default is 512.
        left_crop : int
            left image margin to be cropped off BEFORE distortion correction (via deformation field) is applied. Default is object attribute (or 0 if absent).
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is image attribute (or np.nan if absent - no distortion correction).
            Deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        fill_value : int
            The value to assign to pixels outside the transformed image bounds. Default is -10000.
        flatten_mosaic : boolean
            If True, apply mosaic-level field flattening using parameters from 
            determine_mosaic_flattening_parameters(). Default is False.
        perform_intensity_normalization : boolean
            Default is False. If True and tile_scales attribute is available, perform intensity normalization (tile intensity rescaling).
        use_default_coordinates : bool
            If True, use self.default_tr_matr (nominal tile positions from FirstPixels)
            instead of self.tr_matr (SIFT-refined positions). Default is False.
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        save_snapshot : boolean
            If True, build an image that contains the montage and some data. Default is False.
        snapshot_fname : string
            The name of the image to perform these operations (default is self.fls[layer_id].ravel()[0].replace('0-0-0.dat', 'snapshot.png')).
        thr_min : float
            Lower CDF threshold for determining the minimum data value. Default is 1.0e-3
        thr_max : float
            Upper CDF threshold for determining the maximum data value. Default is 1.0e-3
        nbins : int
            Number of histogram bins for building the PDF and CDF.
        overlay_tile_grid : boolean
            If True (Default), overlays tile grid.
        dpi : int
            DPI. Default is 300.
        save_to_dat : boolean
            If True, saves existing montage into a new .dat file. Default is False. Only works on .dat files at the moment
        dat_fname : string
            The name of the image to perform these operations (default is self.fls[layer_id].ravel()[0].replace('0-0-0.dat', 'layer_mosaic.dat')).
        save_images : boolean
            If True, saves existing mosaic(s) into a .jpg or .png file(s)
        image_fname : string
            The name of the image to perform these operations (default is self.fls[layer_id].ravel()[0].replace('0-0-0.dat', 'layer_mosaic.jpg')).
        verbose : bool
            If True, the intermediate results are displayed. Default is False.
        interpolation : int
            Default is cv2.INTER_LINEAR
        border_value : float
            borderValue for cv2.remap. Default is np.nan
        border_mode : int
            borderMode for cv2.remap. Default is cv2.BORDER_CONSTANT
        bin_factor : int
            Output binning factor (>= 1). When > 1, the assembled layer is
            binned by mean over (bin_factor x bin_factor) blocks before being
            returned. Output shape becomes (Ysize // bin_factor,
            (Xsize - left_crop) // bin_factor). Default is 1 (no binning).

        Returns:
        ----------
        layer_mosaics, layer_id
        
        '''
        if hasattr(self, 'DetB'):
            ifDetB = (self.DetB != 'None')
        else:
            ifDetB = False
        image_names_default = ['RawImageA']
        if ifDetB:
            image_names_default.append('RawImageB')
        image_names = kwargs.get('image_names', image_names_default)
        if layer_id < -1 or layer_id > self.nz_tiles - 1:
            raise ValueError(
                "assemble_layer_mosaic: layer_id={:d} is out of range; "
                "valid range is -1..{:d}".format(layer_id, self.nz_tiles - 1))
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=True)
        left_crop = kwargs.get('left_crop', self.left_crop)
        deformation_field = kwargs.get('deformation_field', self.deformation_field)
        weight_min = kwargs.get('weight_min', 1.0)
        weight_max = kwargs.get('weight_max', 512.0)
        fill_value = kwargs.get('fill_value', -10000) 
        perform_intensity_normalization = kwargs.get('perform_intensity_normalization', False)
        verbose = kwargs.get('verbose', False)
        flatten_mosaic = kwargs.get('flatten_mosaic', False)
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        border_value = kwargs.get('border_value', np.nan)
        border_mode = kwargs.get('border_mode', cv2.BORDER_CONSTANT)
        data_dir = kwargs.get('data_dir', self.data_dir)
        save_snapshot = kwargs.get('save_snapshot', False)
        snapshot_fname = kwargs.get('snapshot_fname',  self.fls[layer_id].ravel()[0].replace('0-0-0.dat', 'snapshot.png'))
        save_to_dat = kwargs.get('save_to_dat', False)
        dat_fname = kwargs.get('dat_fname',  self.fls[layer_id].ravel()[0].replace('0-0-0.dat', 'layer_mosaic.dat'))
        save_images = kwargs.get('save_images', False)
        image_fname = kwargs.get('image_fname',  self.fls[layer_id].ravel()[0].replace('0-0-0.dat', 'layer_mosaic.jpg'))
        overlay_tile_grid = kwargs.get('overlay_tile_grid', True)
        thr_min = kwargs.get('thr_min', 1.0e-3)
        thr_max = kwargs.get('thr_max', 1.0e-3)
        nbins = kwargs.get('nbins', 256)
        linestyle = kwargs.get('linestyle', 'dashed')
        linewidth = kwargs.get('linewidth', 0.25)
        fontsize = kwargs.get('fontsize', 6)
        color = kwargs.get('color', 'cyan')
        dtp = kwargs.get('dtp', np.int16)
        dpi = kwargs.get('dpi', 300)
        use_default_coordinates = kwargs.get('use_default_coordinates', False)
        DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        bin_factor = kwargs.get('bin_factor', 1)
        if not isinstance(bin_factor, int) or bin_factor < 1:
            raise ValueError(
                f"assemble_layer_mosaic: bin_factor must be a positive int (got {bin_factor!r})."
            )

        return_layer_array = True
        save_tif = False
        tif_fname = ''
        layer_mosaics = []

        kwargs_al = {'verbose' : verbose,
                    'interpolation' : interpolation,
                    'border_value' : border_value,
                    'border_mode' : border_mode,
                    'local_DASK_client' : DASK_client,
                    'DASK_client_retries' : DASK_client_retries}

        if use_default_coordinates:
            tr_matr_layer = self.default_tr_matr[layer_id]
        else:
            tr_matr_layer = self.tr_matr[layer_id]
        
        for j, image_name in enumerate(image_names):
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S  ') + ' processing the data for ' + image_name)
            if perform_intensity_normalization:
                params = [layer_id, self.fls[layer_id].ravel(), image_name, tr_matr_layer, weight_min, weight_max,
                          fill_value, self.Xsize, self.Ysize, left_crop,
                          self.tile_I0s[layer_id], self.tile_scales[layer_id],
                          return_layer_array, save_tif, tif_fname, dtp, bin_factor, verbose]
            else:
                params = [layer_id, self.fls[layer_id].ravel(), image_name, tr_matr_layer, weight_min, weight_max,
                          fill_value, self.Xsize, self.Ysize, left_crop,
                          np.zeros(self.n_tiles_per_layer), np.ones(self.n_tiles_per_layer),
                          return_layer_array, save_tif, tif_fname, dtp, bin_factor, verbose]

            # Add per-image flattening parameters
            kwargs_al_local = dict(kwargs_al)
            if flatten_mosaic and hasattr(self, 'mosaic_correction_coeffs'):
                kwargs_al_local['flatten_mosaic'] = True
                kwargs_al_local['mosaic_correction_intercept'] = self.mosaic_correction_intercepts[j]
                kwargs_al_local['mosaic_correction_coeffs'] = self.mosaic_correction_coeffs[j]
                kwargs_al_local['mosaic_correction_degree'] = self.mosaic_correction_degrees[j]
                kwargs_al_local['mosaic_correction_bins'] = self.mosaic_correction_bins
                # Determine offset for Raw images
                if image_name == 'RawImageA':
                    kwargs_al_local['mosaic_Scaling_offset'] = self.Scaling[1, 0]
                elif image_name == 'RawImageB':
                    kwargs_al_local['mosaic_Scaling_offset'] = self.Scaling[1, 1]
                else:
                    kwargs_al_local['mosaic_Scaling_offset'] = 0.0

            layer_mosaics.append(assemble_layer(params, deformation_field, **kwargs_al_local)[0])

        if save_snapshot:
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S  ') + ' saving snapshot')
            if ifDetB:
                try:
                    vminB, vmaxB = get_min_max_thresholds(layer_mosaics[1], thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
                    fig, axs = plt.subplots(3, 1, figsize=(11,8))
                except:
                    ifDetB = False
                    pass
            if not ifDetB:
                fig, axs = plt.subplots(2, 1, figsize=(7,8))
            fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.15, hspace=0.1)
            vminA, vmaxA = get_min_max_thresholds(layer_mosaics[0], thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            axs[1].imshow(layer_mosaics[0], cmap='Greys', vmin=vminA, vmax=vmaxA)
            if ifDetB:
                axs[2].imshow(layer_mosaics[1], cmap='Greys', vmin=vminB, vmax=vmaxB)
            try:
                ttls = [self.Notes.strip('\x00'),
                    'Detector A:  '+ self.DetA.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(vminA, vmaxA, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessA, self.ContrastA),
                    'Detector B:  '+ self.DetB.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(vminB, vmaxB, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessB, self.ContrastB)]
            except:
                ttls = ['', 'Detector A', '']
            for j, ax in enumerate(axs):
                ax.axis(False)
                ax.set_title(ttls[j], fontsize=10)
                if overlay_tile_grid:
                        overlay_montage_grid(ax, self,
                                tile_positions = -tr_matr_layer[:, 0:2, 2],
                                bin_factor = bin_factor,
                                 linewidth = linewidth,
                                 linestyle = linestyle,
                                 color = color)
            fig.suptitle(snapshot_fname, fontsize = fontsize)

            if hasattr(self, 'EHT'):
                EHT_text = '{:.3f} kV'.format(self.EHT)
            else:
                EHT_text = ''
            if hasattr(self, 'SEMCurr'):
                SEMCurr_text = '{:.3f} nA'.format(self.SEMCurr*1.0e9)
            else:
                SEMCurr_text = ''
            if hasattr(self, 'ScanRate'):
                ScanRate_text = '{:.3f} MHz'.format(self.ScanRate/1.0e6)
            else:
                ScanRate_text = ''
            if hasattr(self, 'WD'):
                WD_text = '{:.3f} mm'.format(self.WD)
            else:
                WD_text = ''
            if hasattr(self, 'MachineID'):
                MachineID_text = '{:s}'.format(self.MachineID.strip('\x00'))
            else:
                MachineID_text = ''
            if hasattr(self, 'Sample_ID'):
                Sample_ID_text = self.Sample_ID.strip('\x00')
            else:
                Sample_ID_text = ''
            if hasattr(self, 'shape'):
                shape_strings = 'Tile Size\n\nShape', '{:d} x {:d}\n\n{:d} x {:d}'.format(self.XResolution, self.YResolution, self.shape[1], self.shape[0]), '',
            else:
                shape_strings = 'Tile Size\n\n # of Tiles per layer', '{:d} x {:d}\n\n{:d}'.format(self.XResolution, self.YResolution, self.n_tiles_per_layer), ''
            if self.FileVersion > 8:
                cell_text = [['Sample ID', '{:s}'.format(self.Sample_ID.strip('\x00')), '',
                              *shape_strings,
                              'Scan Rate', '{:.3f} MHz'.format(self.ScanRate/1.0e6)],
                            ['Machine ID', '{:s}'.format(self.MachineID.strip('\x00')), '',
                              'Pixel Size', '{:.1f} nm'.format(self.PixelSize), '',
                              'Oversampling', '{:d}'.format(self.Oversampling)],
                             ['FileVersion', '{:d}'.format(self.FileVersion), '',
                              'Working Dist.', '{:.3f} mm'.format(self.WD), '',
                              'FIB Focus', '{:.1f}  V'.format(self.FIBFocus)],
                             ['Bit Depth', '{:d}'.format(8 *(2 - self.EightBit)), '',
                             'EHT Voltage\n\nSEM Current', '{:.3f} kV \n\n{:.3f} nA'.format(self.EHT, self.SEMCurr*1.0e9), '',
                             'FIB Probe', '{:d}'.format(self.FIBProb)]]
            else:
                if self.FileVersion > 0:
                    cell_text = [['', '', '',
                                  *shape_strings,
                                  'Scan Rate', '{:.3f} MHz'.format(self.ScanRate/1.0e6)],
                                ['Machine ID', '{:s}'.format(self.MachineID.strip('\x00')), '',
                                  'Pixel Size', '{:.1f} nm'.format(self.PixelSize), '',
                                  'Oversampling', '{:d}'.format(self.Oversampling)],
                                 ['FileVersion', '{:d}'.format(self.FileVersion), '',
                                  'Working Dist.', '{:.3f} mm'.format(self.WD), '',
                                  'FIB Focus', '{:.1f}  V'.format(self.FIBFocus)],
                                 ['Bit Depth', '{:d}'.format(8 *(2 - self.EightBit)), '',
                                 'EHT Voltage', '{:.3f} kV'.format(self.EHT), '',
                                 'FIB Probe', '{:d}'.format(self.FIBProb)]]
                else:
                    cell_text = [['Sample ID', '{:s}'.format(Sample_ID_text), '',
                                  *shape_strings,
                                  'Scan Rate', ScanRate_text],
                                ['Machine ID', MachineID_text, '',
                                  'Pixel Size', '{:.1f} nm'.format(self.PixelSize), '',
                                  'Oversampling', ''],
                                 ['FileVersion', '{:d}'.format(self.FileVersion), '',
                                  'Working Dist.', WD_text, '',
                                  'FIB Focus', ''],
                                 ['Bit Depth', '{:d}'.format(8 *(2 - self.EightBit)), '',
                                 'EHT Voltage\n\nSEM Current', EHT_text+' \n\n'+SEMCurr_text, '',
                                 'FIB Probe', '']]
            llw0=0.3
            llw1=0.18
            llw2=0.02
            clw = [llw1, llw0, llw2, llw1, llw1, llw2, llw1, llw1]
            axs[0].axis(False)
            tbl = axs[0].table(cellText=cell_text,
                               colWidths=clw,
                               cellLoc='center',
                               colLoc='center',
                               bbox = [0.02, 0, 0.96, 1.0],
                               #bbox = [0.45, 1.02, 2.8, 0.55],
                               zorder=10)
            fig.savefig(snapshot_fname, dpi=dpi)
            display(fig)
            plt.close(fig)
        
        if save_to_dat:
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S  ') + ' saving into .dat file')
            '''
            Save existing montage into a new .dat file. Only works on .dat files at the moment. ©G.Shtengel 11/2025 gleb.shtengel@gmail.com
            '''
            YResolution_new = self.Ysize
            XResolution_new = self.Xsize - left_crop
            if verbose:
                print('Output Frame Size: {:d} x {:d} pixels'.format(XResolution_new, YResolution_new))
                print('Data will be saved into the file: ', dat_fname)
                print('Will use fill value = {:d}'.format(fill_value))

            # Update the header with new frame size information
            fr = FIBSEM_frame(self.fls[layer_id].ravel()[0], read_header_only=True)
            header = fr.header
            header_new = bytearray(header)
            XResolution_new_string =  pack('>L', XResolution_new)
            header_new[100:104] = XResolution_new_string
            YResolution_new_string =  pack('>L', YResolution_new)
            header_new[104:108] = YResolution_new_string

            # Create new Raw data array
            dt = np.dtype(np.int16).newbyteorder('>')

            # Save new frame
            with open(dat_fname, 'wb') as f:
                f.write(header_new)
                np.moveaxis(np.array(layer_mosaics), 0, 2).astype(dt).tofile(f)

        if save_images:
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S  ') + ' saving images into .jpg files')
            imf1, imf2 = os.path.splitext(image_fname)
            for j, layer_mosaic in enumerate(layer_mosaics):
                siy, six = layer_mosaic.shape
                s = max((six, siy))
                fig, ax = plt.subplots(1,1, figsize=(10.0*six/s,10.0*siy/s))
                fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
                if j == 0:
                    if not save_snapshot:
                        vmin, vmax = get_min_max_thresholds(layer_mosaic, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
                    else:
                        vmin = vminA
                        vmax = vmaxA
                    try:
                        det_str = 'Detector A:  '+ self.DetA.strip('\x00')
                    except:
                        det_str = 'Detector:'
                else:
                    if not save_snapshot:
                        vmin, vmax = get_min_max_thresholds(layer_mosaic, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
                    else:
                        vmin = vminB
                        vmax = vmaxB
                    try:
                        det_str = 'Detector B:  '+ self.DetB.strip('\x00')
                    except:
                        det_str = 'Detector:'
                print(det_str + ', data range: vmin={:.2f}, vmax={:.2f}'.format(vmin, vmax))
                ax.imshow(layer_mosaic, cmap='Greys', vmin = vmin, vmax = vmax)
                ax.axis(False)
                overlay_montage_grid(ax, self,
                                     tile_positions = -tr_matr_layer[:, 0:2, 2],
                                     bin_factor = bin_factor,
                                     left_crop = left_crop,
                                     linewidth = linewidth,
                                     linestyle = linestyle,
                                     color = color)
                if j == 1:
                    image_fname_loc = imf1 + '_' + self.DetB.strip('\x00') + imf2
                else:
                    if hasattr(self, 'DetA'):
                        image_fname_loc = imf1 + '_' + self.DetA.strip('\x00') + imf2
                    else:
                        image_fname_loc = image_fname
                fig.savefig(image_fname_loc, dpi=dpi)
                display(fig)
                plt.close(fig)

        return layer_mosaics, layer_id


    def determine_mosaic_flattening_parameters(self, **kwargs):
        '''
        Perform 2D polynomial fit on assembled mosaic layer(s) and determine 
        the field-flattening parameters.
        Calls Perform_2D_fit(img, estimator, **kwargs) for each mosaic image.

        kwargs:
        ----------
        layer_mosaics : list of 2D arrays
            Pre-assembled mosaic images (from assemble_layer_mosaic).
            If not provided, layer_id must be given and the mosaic will be assembled.
        layer_id : int
            Layer ID for assembling the mosaic via assemble_layer_mosaic.
            Used only if layer_mosaics is not provided. Default is 0.
        image_names : list of str
            Image source names, must match the order of layer_mosaics.
            Default is ['RawImageA', 'RawImageB'] if DetB available, ['RawImageA'] otherwise.
        estimator : sklearn estimator
            Default is LinearRegression().
        bins : int
            Binning size (in pixel units) for image binning. Default is 10.
        degrees : int or list of int
            Polynomial degree(s). Default is 2.
        Analysis_ROIs : list of lists: [[left, right, top, bottom]]
        save_correction_binary : boolean
            Save the image_name and img_correction_array data into a binary file. Default is False.
        ignore_Y : boolean
        linear_Y : boolean
        Xsect : int
        Ysect : int
        disp_res : boolean
        verbose : boolean
        save_res_png : boolean
        res_fname : string
        dpi : int
        
        Returns:
        ----------
        mosaic_correction_intercepts, mosaic_correction_coeffs
        '''
        # --- fitting parameters ---
        estimator = kwargs.get("estimator", LinearRegression())
        bins = kwargs.get("bins", 10)
        degrees = kwargs.get("degrees", 2)
        ignore_Y = kwargs.get("ignore_Y", False)
        linear_Y = kwargs.get("linear_Y", False)
        disp_res = kwargs.get("disp_res", True)
        verbose = kwargs.get("verbose", True)
        Analysis_ROIs = kwargs.get("Analysis_ROIs", [])
        save_correction_binary = kwargs.get("save_correction_binary", False)
        save_res_png = kwargs.get("save_res_png", False)
        res_fname = kwargs.get("res_fname", '_Mosaic_Image_Flattening.png')
        dpi = kwargs.get("dpi", 300)

        # --- resolve image_names (same pattern as assemble_layer_mosaic) ---
        if hasattr(self, 'DetB'):
            ifDetB = (self.DetB != 'None')
        else:
            ifDetB = False
        image_names_default = ['RawImageA']
        if ifDetB:
            image_names_default.append('RawImageB')
        image_names = kwargs.get('image_names', image_names_default)

        # --- get or assemble mosaics ---
        if 'layer_mosaics' in kwargs:
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S  ') + ' using existing layer_mosaics to determine flattening parameters')
            layer_mosaics = kwargs['layer_mosaics']
        else:
            layer_id = kwargs.get('layer_id', 0)
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S  ') + ' no layer_mosaics provided')
                print('Will build the layer_mosaics for layer_id={:d} to determine flattening parameters'.format(layer_id))
            layer_mosaics, _ = self.assemble_layer_mosaic(layer_id, **kwargs)


        mosaic_correction_coeffs = []
        mosaic_correction_intercepts = []
        mosaic_correction_degrees = []
        mosaic_correction_arrays = []

        for j, (image_name, mosaic) in enumerate(zip(image_names, layer_mosaics)):
            # subtract offset for Raw images (same as FIBSEM_frame version)
            if image_name == 'RawImageA':
                img = mosaic - self.Scaling[1, 0]
            elif image_name == 'RawImageB':
                img = mosaic - self.Scaling[1, 1]
            else:
                img = mosaic

            ysz, xsz = img.shape
            Xsect = kwargs.get("Xsect", xsz // 2)
            Ysect = kwargs.get("Ysect", ysz // 2)

            Fit_kwargs = {'image_name': image_name,
                          'calc_corr': save_correction_binary,
                          'ignore_Y': ignore_Y,
                          'linear_Y': linear_Y,
                          'Xsect': Xsect,
                          'Ysect': Ysect,
                          'disp_res': disp_res,
                          'bins': bins,
                          'Analysis_ROIs': Analysis_ROIs,
                          'save_res_png': save_res_png,
                          'res_fname': res_fname.replace('.png', '_' + image_name + '.png'),
                          'dpi': dpi}
            try:
                Fit_kwargs['degree'] = degrees[j]
            except:
                Fit_kwargs['degree'] = degrees

            intercept, coeffs, mse, mosaic_correction_array = Perform_2D_fit(img, estimator, **Fit_kwargs)
            mosaic_correction_coeffs.append(coeffs)
            mosaic_correction_intercepts.append(intercept)
            mosaic_correction_degrees.append(Fit_kwargs['degree'])
            mosaic_correction_arrays.append(mosaic_correction_array)

        self.mosaic_correction_sources = image_names
        self.mosaic_correction_coeffs = mosaic_correction_coeffs
        self.mosaic_correction_intercepts = mosaic_correction_intercepts
        self.mosaic_correction_degrees = mosaic_correction_degrees
        self.mosaic_correction_bins = bins

        if save_correction_binary:
            bin_fname = res_fname.replace('png', 'bin')
            pickle.dump([image_names, mosaic_correction_arrays], open(bin_fname, 'wb')) # saves source name and correction array into the binary file
            self.image_correction_file = bin_fname
            print('Image Flattening Info saved into the binary file: ', self.image_correction_file)

        return mosaic_correction_intercepts, mosaic_correction_coeffs


    def flatten_layer_mosaic(self, layer_mosaics, **kwargs):
        '''
        Flatten assembled mosaic layer(s) using stored polynomial coefficients.
        Calls the standalone function flatten_image_fast(img, intercept, coeffs, degree, bins)
        for each mosaic image.

        Parameters:
        ----------
        layer_mosaics : list of 2D arrays
            Assembled mosaic images (from assemble_layer_mosaic).

        kwargs:
        ----------
        mosaic_correction_sources : list of str
        mosaic_correction_intercepts : list of float
        mosaic_correction_coeffs : list of 1D arrays
        mosaic_correction_degrees : list of int
        mosaic_correction_bins : int

        Returns:
        ----------
        flattened_mosaics : list of 2D arrays
        '''
        mosaic_correction_sources = kwargs.get("mosaic_correction_sources",
            getattr(self, 'mosaic_correction_sources', [False]))
        mosaic_correction_intercepts = kwargs.get("mosaic_correction_intercepts",
            getattr(self, 'mosaic_correction_intercepts', [False]))
        mosaic_correction_coeffs = kwargs.get("mosaic_correction_coeffs",
            getattr(self, 'mosaic_correction_coeffs', [False]))
        mosaic_correction_degrees = kwargs.get("mosaic_correction_degrees",
            getattr(self, 'mosaic_correction_degrees', [2]))
        bins = kwargs.get("mosaic_correction_bins",
            getattr(self, 'mosaic_correction_bins', 10))

        flattened_mosaics = []
        for mosaic, source, intercept, coeffs, degree in zip(
                layer_mosaics,
                mosaic_correction_sources,
                mosaic_correction_intercepts,
                mosaic_correction_coeffs,
                mosaic_correction_degrees):

            if (source is not False) and (coeffs is not False):
                if source == 'RawImageA':
                    img = mosaic - self.Scaling[1, 0]
                    flattened_mosaic = flatten_image_fast(img, intercept, coeffs, degree, bins) + self.Scaling[1, 0]
                elif source == 'RawImageB':
                    img = mosaic - self.Scaling[1, 1]
                    flattened_mosaic = flatten_image_fast(img, intercept, coeffs, degree, bins) + self.Scaling[1, 1]
                else:
                    flattened_mosaic = flatten_image_fast(mosaic, intercept, coeffs, degree, bins)
            else:
                flattened_mosaic = mosaic

            flattened_mosaics.append(flattened_mosaic)

        return flattened_mosaics


    def save_stack(self, **kwargs):
        '''
        Assemble all layers based on transformation matrices for each tile and save them into stack. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        fnm_mosaic_stack : string
            Filename to save the data. Default is object attribute self.fnm_mosaic_stack
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['mrc', 'tifs'].
            Default is ['mrc']. If 'tifs' is selected, data is saved as individual tif files,
            one per layer. Use empty list if do not want to save the data.
        image_name : str
            Image name ('RawImageA' or 'RawImageB'). Default is 'RawImageA'.
        flatten_mosaic : boolean
            If True, apply mosaic-level field flattening using parameters from
            determine_mosaic_flattening_parameters(). Default is False.
        tif_folder : str
            sub-directory name (will be created inside data_dir). Default is 'tif_stack'
        voxel_size : rec array of 3 elements
            voxel size in nm
        dtp  : dtype
            Python data type for saving. Default is int16, the other option currently is uint8.
        U8_range : list [U8_min, U8_max]
            Optional conversion range for uint8 output. Only used when dtp=np.uint8.
            Data is clipped to [U8_min, U8_max] and rescaled to [0, 255] before casting.
            Default is None (plain cast, data must already be in [0, 255]).
        weight_min : float
            vmin for weight. Default is 1
        weight_max : float
            vmax for weight. Default is 512
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_field or on its own). Default is 0 - no cropping.
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction.
        fill_value : int
            The value to assign to pixels outside the transformed image bounds. Default is -10000.
        perform_intensity_normalization : boolean
            Default is False. If True and tile_scales attribute is available, perform intensity normalization (tile intensity rescaling).
        use_default_coordinates : bool
            If True, use self.default_tr_matr (nominal tile positions from FirstPixels)
            instead of self.tr_matr (SIFT-refined positions). Default is False.
        verbose : bool
            If True, the intermediate results are displayed. Default is False.
        interpolation : int
            Default is cv2.INTER_LINEAR
        border_value : float
            borderValue for cv2.remap. Default is np.nan
        border_mode : int
            borderMode for cv2.remap. Default is cv2.BORDER_CONSTANT
        
        Returns:
        ----------
        fnms_saved
        
        '''
        perform_intensity_normalization = kwargs.get('perform_intensity_normalization', False)
        use_default_coordinates = kwargs.get('use_default_coordinates', False)
        DASK_client = kwargs.get('DASK_client', '')
        dtp = kwargs.get("dtp", np.int16)
        U8_range = kwargs.get('U8_range', None)
        image_name = kwargs.get('image_name', 'RawImageA')
        weight_min = kwargs.get('weight_min', 1.0)
        kwargs['weight_min'] = weight_min 
        weight_max = kwargs.get('weight_max', 512.0)
        kwargs['weight_max'] = weight_max 
        fill_value = kwargs.get('fill_value', -10000)
        kwargs['fill_value'] = fill_value
        verbose = kwargs.get('verbose', False)
        flatten_mosaic = kwargs.get('flatten_mosaic', False)
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        border_value = kwargs.get('border_value', np.nan)
        border_mode = kwargs.get('border_mode', cv2.BORDER_CONSTANT)
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=True)
        left_crop = kwargs.get('left_crop', self.left_crop)
        deformation_field = kwargs.get('deformation_field', self.deformation_field)
        kwargs['left_crop'] = left_crop
        fnm_mosaic_stack = kwargs.get('fnm_mosaic_stack', self.fnm_mosaic_stack)
        fnm_types = kwargs.get("fnm_types", ['mrc'])
        allowed_fnm_types = {'mrc', 'tifs'}
        invalid_fnm_types = set(fnm_types) - allowed_fnm_types
        fnms_saved = []
        if invalid_fnm_types:
            raise ValueError(f"save_stack: invalid fnm_types value(s): {sorted(invalid_fnm_types)}. "
                f"Allowed values are: {sorted(allowed_fnm_types)}")
        tif_folder = kwargs.get('tif_folder', 'tif_stack')
        
        save_tif = False
        save_folder = ''
        if 'tifs' in fnm_types:
            save_tif = True
            save_folder = os.path.join(os.path.split(fnm_mosaic_stack)[0], tif_folder)
            os.makedirs(save_folder, exist_ok=True)
        
        save_mrc = False
        return_layer_array = False
        if 'mrc' in fnm_types:
            save_mrc = True
            return_layer_array = True
            '''
            mode 0 -> uint8
            mode 1 -> int16
            mode 6 -> uint16
            '''
            mrc_mode = 0
            if dtp==np.int16:
                mrc_mode = 1
            if dtp==np.uint16:
                mrc_mode = 6
            if dtp==np.float16:
                dtp=np.int16
                mrc_mode = 1
        
        voxel_size = kwargs.get("voxel_size", self.voxel_size)
        voxel_size_angstr = voxel_size.copy()
        voxel_size_angstr.x = voxel_size_angstr.x * 10.0
        voxel_size_angstr.y = voxel_size_angstr.y * 10.0
        voxel_size_angstr.z = voxel_size_angstr.z * 10.0

        DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        kwargs_al = {'verbose' : verbose,
            'interpolation' : interpolation,
            'border_value' : border_value,
            'border_mode' : border_mode}
        if U8_range is not None:
            kwargs_al['U8_range'] = U8_range

        if flatten_mosaic and hasattr(self, 'mosaic_correction_coeffs'):
            # Find the index of image_name in mosaic_correction_sources
            try:
                corr_idx = self.mosaic_correction_sources.index(image_name)
                kwargs_al['flatten_mosaic'] = True
                kwargs_al['mosaic_correction_intercept'] = self.mosaic_correction_intercepts[corr_idx]
                kwargs_al['mosaic_correction_coeffs'] = self.mosaic_correction_coeffs[corr_idx]
                kwargs_al['mosaic_correction_degree'] = self.mosaic_correction_degrees[corr_idx]
                kwargs_al['mosaic_correction_bins'] = self.mosaic_correction_bins
                if image_name == 'RawImageA':
                    kwargs_al['mosaic_Scaling_offset'] = self.Scaling[1, 0]
                elif image_name == 'RawImageB':
                    kwargs_al['mosaic_Scaling_offset'] = self.Scaling[1, 1]
                else:
                    kwargs_al['mosaic_Scaling_offset'] = 0.0
            except ValueError:
                print('Warning: no mosaic flattening parameters found for ' + image_name)
                flatten_mosaic = False

        if fnm_types:            
            if save_mrc:
                mrc_filename = os.path.splitext(fnm_mosaic_stack)[0] + '.mrc'
                fnms_saved.append(mrc_filename)
                stack_shape = (self.nz_tiles, self.Ysize, self.Xsize-left_crop)
                mrc_new = mrcfile.new_mmap(mrc_filename, shape = stack_shape, mrc_mode=mrc_mode, overwrite=True)
                mrc_new.voxel_size = voxel_size_angstr
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the registered stack into the file: ', mrc_filename)
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Stack dimensions nz, ny, nx (pixels): {:d} x {:d} x {:d}'.format(*stack_shape))
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Stack Voxel Size (Angstroms): {:.2f} x {:.2f} x {:.2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))

            layer_ids = np.arange(self.nz_tiles)
            params_mult = []
            for layer_id in layer_ids:
                fls_layer = self.fls[layer_id].ravel()
                if use_default_coordinates:
                    tr_matr_layer = self.default_tr_matr[layer_id]
                else:
                    tr_matr_layer = self.tr_matr[layer_id]
                tif_fname = os.path.join(save_folder, os.path.splitext(os.path.split(fnm_mosaic_stack)[1])[0] + '_layer_{:06d}.tif'.format(layer_id))
                if save_tif:
                    fnms_saved.append(tif_fname)
                if perform_intensity_normalization:
                    params_mult.append([layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max,
                                        fill_value, self.Xsize, self.Ysize, left_crop,
                                        self.tile_I0s[layer_id], self.tile_scales[layer_id],
                                        return_layer_array, save_tif, tif_fname, dtp, 1, verbose])  # bin_factor = 1
                else:
                    params_mult.append([layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max,
                                        fill_value, self.Xsize, self.Ysize, left_crop,
                                        np.zeros(len(fls_layer)), np.ones(len(fls_layer)),
                                        return_layer_array, save_tif, tif_fname, dtp, 1, verbose])  # bin_factor = 1
            if use_DASK:
                shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                futures = DASK_client.map(assemble_layer, params_mult, deformation_field = shared_data_future, retries = DASK_client_retries, **kwargs_al)
                for future in tqdm(as_completed(futures), total=len(futures), desc='Assembling and saving mosaic layers'):
                    mosaic_out, j = future.result()
                    if save_mrc:
                        if dtp == np.uint8 and U8_range is not None:
                            U8_min, U8_max = float(U8_range[0]), float(U8_range[1])
                            scale = 255.0 / max(U8_max - U8_min, 1e-6)
                            mrc_new.data[j, :, :] = np.clip((mosaic_out - U8_min) * scale, 0, 255).astype(np.uint8)
                        else:
                            mrc_new.data[j, :, :] = mosaic_out
                    future.cancel()
            else:
                for j, params in enumerate(tqdm(params_mult, desc = 'Assembling and saving mosaic layers')):
                    mosaic_out = assemble_layer(params, deformation_field, **kwargs_al)[0]
                    if save_mrc:
                        if dtp == np.uint8 and U8_range is not None:
                            U8_min, U8_max = float(U8_range[0]), float(U8_range[1])
                            scale = 255.0 / max(U8_max - U8_min, 1e-6)
                            mrc_new.data[j, :, :] = np.clip((mosaic_out - U8_min) * scale, 0, 255).astype(np.uint8)
                        else:
                            mrc_new.data[j, :, :] = mosaic_out
            if save_mrc:
                mrc_new.close()
            
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving Finished')
        return fnms_saved


    def save_stack_zarr3(self, **kwargs):
        '''
        One-stop save of the assembled mosaic stack directly into a sharded ZARR v3
        store (OME-NGFF v0.4 multiscales metadata). No intermediate MRC/v2 store.
        ©G.Shtengel 2026 gleb.shtengel@gmail.com

        Each DASK worker handles ONE shard. At s0, the worker composites the
        contributing tiles (via transform_tile + _add_warped_to_mosaic) into a
        shard-local buffer and writes the whole shard atomically.  For pyramid
        levels s1, s2, …, each worker reads one shard's worth of source from
        the previous level, downsamples by `downsample_factor`, and writes the
        whole shard.

        kwargs (mirrors save_stack where applicable):
        ----------
        DASK_client            : DASK client; '' for local (default '').
        DASK_client_retries    : int.  Default self.DASK_client_retries.
        max_futures            : int.  Default self.max_futures (50000).
            Max DASK futures per batch. Caps driver memory and scheduler graph size
            when writing very large stacks (100 TB scale: ~820k s0 shards).
        output_zarr_path       : str.  Default = splitext(self.fnm_mosaic_stack)[0] + '.zarr'.
        image_name             : 'RawImageA' | 'RawImageB'.  Default 'RawImageA'.
        weight_min, weight_max : floats.  Defaults 1.0, 512.0.
        fill_value             : float.  Default -10000.
        left_crop              : int.    Default self.left_crop.
        deformation_field      : 3D array.  Default self.deformation_field.
        perform_intensity_normalization : bool.  Default False.
        use_default_coordinates : bool.  Default False.
        flatten_mosaic         : bool.  Default False.
        dtp                    : numpy dtype.  Default int16.
        U8_range               : [umin, umax] for uint8 output.  Default None.
        verbose                : bool.  Default False.
        interpolation, border_value, border_mode : cv2 args.

        # v3-specific (defaults follow convert_ome_zarr_v2_to_v3 / tif_stack_to_zarr3)
        chunk_size             : tuple, default (32, 32, 32) in axis_order.
        shard_size             : tuple, default (1024, 1024, 1024) in axis_order.
        axis_order             : str,   default 'xyz' (source is always ZYX).
        transpose_codec        : bool,  default True (F-order inside chunks).
        compression            : str,   default 'zstd'.
        compression_level      : int,   default 3.
        n_pyramid_levels       : int,   default 4.
        downsample_factor      : int,   default 2.
        voxel_unit             : str,   default 'nanometer'.
        zarr_origin_zyx        : tuple, default (0.0, 0.0, 0.0).
        zarr_dataset_name      : str,   default 'volume'.

        # Neuroglancer
        neuroglancer_serve_base_url, neuroglancer_viewer_url,
        neuroglancer_display_axes_order : forwarded to generate_neuroglancer_link.

        Returns:
        ----------
        dict
            {'output_zarr_path', 'shape_zyx', 'shape_out', 'dtype',
             'chunks', 'shards', 'n_levels', 'elapsed_s', 'neuroglancer_link'}
        '''
        import math, itertools, time
        import numpy as np
        import zarr
        from FIBSEM_gs_py.tif_stack_to_zarr import (
            _make_v3_compressor, _resolve_chunks_shards, _translation_at_level,
            generate_neuroglancer_link, _print_neuroglancer_info,
        )

        # ---- 0. Read kwargs ----------------------------------------------
        verbose = kwargs.get('verbose', False)
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, _ = check_DASK(DASK_client, verbose=True)
        DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        max_futures = kwargs.get('max_futures', self.max_futures)
        image_name = kwargs.get('image_name', 'RawImageA')
        weight_min = kwargs.get('weight_min', 1.0)
        weight_max = kwargs.get('weight_max', 512.0)
        fill_value = kwargs.get('fill_value', -10000)
        left_crop  = kwargs.get('left_crop', self.left_crop)
        deformation_field = kwargs.get('deformation_field', self.deformation_field)
        perform_norm = kwargs.get('perform_intensity_normalization', False)
        use_default_coords = kwargs.get('use_default_coordinates', False)
        flatten_mosaic = kwargs.get('flatten_mosaic', False)
        dtp = kwargs.get('dtp', np.int16)
        U8_range = kwargs.get('U8_range', None)
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        border_value  = kwargs.get('border_value', np.nan)
        border_mode   = kwargs.get('border_mode', cv2.BORDER_CONSTANT)

        fnm_mosaic_stack = kwargs.get('fnm_mosaic_stack', self.fnm_mosaic_stack)
        output_zarr_path = kwargs.get('output_zarr_path',
                                      os.path.splitext(fnm_mosaic_stack)[0] + '.zarr')

        # v3 kwargs
        chunk_size  = kwargs.get('chunk_size', (32, 32, 32))
        shard_size  = kwargs.get('shard_size', (1024, 1024, 1024))
        axis_order  = kwargs.get('axis_order', 'xyz').lower()
        transpose_codec   = kwargs.get('transpose_codec', True)
        compression       = kwargs.get('compression', 'zstd')
        compression_level = kwargs.get('compression_level', 3)
        n_pyramid_levels  = kwargs.get('n_pyramid_levels', 4)
        downsample_factor = kwargs.get('downsample_factor', 2)
        voxel_unit        = kwargs.get('voxel_unit', 'nanometer')
        origin_zyx        = kwargs.get('zarr_origin_zyx', (0.0, 0.0, 0.0))
        dataset_name      = kwargs.get('zarr_dataset_name', 'volume')

        ng_serve_url  = kwargs.get('neuroglancer_serve_base_url', 'https://s3.janelia.org/hess-lab/FIBSEM')
        ng_viewer_url = kwargs.get('neuroglancer_viewer_url', 'https://neuroglancer-demo.appspot.com/')
        ng_axes_order = kwargs.get('neuroglancer_display_axes_order', list(axis_order))

        t_start = time.time()

        # ---- 1. Geometry --------------------------------------------------
        nz = int(self.nz_tiles)
        ny = int(self.Ysize)
        nx = int(self.Xsize - left_crop)
        src_shape_zyx = (nz, ny, nx)

        if set(axis_order) != set('xyz') or len(axis_order) != 3:
            raise ValueError(f"axis_order must be a permutation of 'xyz', got {axis_order!r}")
        axis_perm = tuple('zyx'.index(a) for a in axis_order)   # ZYX -> output
        dst_shape = tuple(src_shape_zyx[axis_perm[i]] for i in range(3))

        # voxel_size_zyx (tuple) takes precedence over voxel_size (rec array).
        # Fallback to self.voxel_size when neither kwarg is supplied.
        vs_kwarg = kwargs.get('voxel_size_zyx', None)
        if vs_kwarg is not None:
            voxel_size_zyx = (float(vs_kwarg[0]), float(vs_kwarg[1]), float(vs_kwarg[2]))
        else:
            voxel_size = kwargs.get('voxel_size', self.voxel_size)
            voxel_size_zyx = (float(voxel_size.z), float(voxel_size.y), float(voxel_size.x))

        # Validate: every component must be positive finite (Neuroglancer requirement).
        if not all(np.isfinite(v) and v > 0 for v in voxel_size_zyx):
            raise ValueError(
                f"save_stack_zarr3: voxel_size_zyx contains non-positive or non-finite "
                f"values: {voxel_size_zyx}. Pass voxel_size_zyx=(zsz, ysz, xsz) as a "
                f"kwarg, or run evaluate_FIBSEM_statistics first to populate "
                f"self.voxel_size from milling-rate data."
            )

        voxel_size_out = tuple(voxel_size_zyx[axis_perm[i]] for i in range(3))
        origin_out     = tuple(origin_zyx[axis_perm[i]]      for i in range(3))
        if verbose:
            print(f"The output stack (axis_order='{axis_order}') will have voxel size: {voxel_size_out}")

        # ---- 2. Pre-allocate v3 store with all pyramid levels -------------
        if os.path.exists(output_zarr_path):
            shutil.rmtree(output_zarr_path)
        root = zarr.open_group(output_zarr_path, mode='w', zarr_format=3)

        v3_compressor = _make_v3_compressor(compression, compression_level)
        transpose_filter = None
        if transpose_codec:
            from zarr.codecs import TransposeCodec
            transpose_filter = TransposeCodec(order=(2, 1, 0))   # F-order, 3D

        level_shapes, level_chunks, level_shards = [], [], []
        cur = dst_shape
        for lvl in range(n_pyramid_levels):
            use_chunks, use_shards = _resolve_chunks_shards(cur, chunk_size, shard_size)
            codec_kwargs = {}
            if transpose_filter is not None: codec_kwargs['filters']     = [transpose_filter]
            if v3_compressor   is not None: codec_kwargs['compressors'] = [v3_compressor]
            root.create_array(
                name=f's{lvl}', shape=cur, dtype=dtp,
                chunks=use_chunks, shards=use_shards, fill_value=0, overwrite=True,
                **codec_kwargs,
            )
            level_shapes.append(cur)
            level_chunks.append(use_chunks)
            level_shards.append(use_shards)
            if verbose:
                print(f"  Pre-allocated s{lvl}  shape={cur}  chunks={use_chunks}  shards={use_shards}")
            cur = tuple(max(d // downsample_factor, 1) for d in cur)

        # ---- 3. OME-NGFF multiscales metadata -----------------------------
        # Per-level translation compensates for the half-voxel offset that
        # appears when level n averages 2^n full-resolution voxels. Use the
        # shared helper so the metadata matches what convert_ome_zarr_v2_to_v3
        # produces for the same data.
        axes_meta = [{'name': a, 'type': 'space', 'unit': voxel_unit} for a in axis_order]
        datasets_meta = []
        for lvl in range(n_pyramid_levels):
            scale = [voxel_size_out[i] * (downsample_factor ** lvl) for i in range(3)]
            translation = _translation_at_level(
                voxel_size_out, origin_out, lvl, downsample_factor,
            )
            datasets_meta.append({
                'path': f's{lvl}',
                'coordinateTransformations': [
                    {'type': 'scale',       'scale':       scale},
                    {'type': 'translation', 'translation': list(translation)},
                ],
            })
        root.attrs['multiscales'] = [{
            'version': '0.4', 'name': dataset_name,
            'axes': axes_meta, 'datasets': datasets_meta,
            'type': 'mean',
            'metadata': {'description': f'save_stack_zarr3 (axis_order={axis_order})'},
        }]

        # ---- 4. Build s0 shard task list ---------------------------------
        # Choose tr_matr source (post-solve or nominal).
        tr_matr_all = self.default_tr_matr if use_default_coords else self.tr_matr

        # Vectorised tile-bbox table for fast shard-membership tests.
        # tile_tx[layer, tile] = -round(tr_matr[layer, tile, 0, 2]) — matches
        # split_translation_int_fract sign convention used by transform_tile.
        tile_tx = -np.round(tr_matr_all[..., 0, 2]).astype(np.int64)   # (nz, n_tiles)
        tile_ty = -np.round(tr_matr_all[..., 1, 2]).astype(np.int64)
        tile_w  = self.XResolution - left_crop
        tile_h  = self.YResolution
        tile_tx1 = tile_tx + tile_w     # exclusive right edge
        tile_ty1 = tile_ty + tile_h     # exclusive bottom edge

        use_s0 = level_shards[0]
        shard_size_xyz = use_s0       # (sx, sy, sz) in output axis order...
        # ...but we enumerate shards in OUTPUT space and convert to ZYX for the worker.
        s0_shape_out = level_shapes[0]
        s0_origins = list(itertools.product(
            *[list(range(0, s0_shape_out[d], use_s0[d])) for d in range(3)]
        ))

        # tile_scales / tile_I0s (broadcasting unity if not present / not requested).
        tile_I0s_all    = self.tile_I0s    if perform_norm else np.zeros_like(self.tile_I0s)
        tile_scales_all = self.tile_scales if perform_norm else np.ones_like(self.tile_scales)

        # flatten_kwargs (per-image), if requested.
        flatten_kwargs_global = None
        if flatten_mosaic and hasattr(self, 'mosaic_correction_coeffs'):
            try:
                corr_idx = self.mosaic_correction_sources.index(image_name)
                flatten_kwargs_global = {
                    'mosaic_correction_intercept': self.mosaic_correction_intercepts[corr_idx],
                    'mosaic_correction_coeffs':    self.mosaic_correction_coeffs[corr_idx],
                    'mosaic_correction_degree':    self.mosaic_correction_degrees[corr_idx],
                    'mosaic_correction_bins':      self.mosaic_correction_bins,
                    'mosaic_Scaling_offset':       (self.Scaling[1, 0] if image_name == 'RawImageA'
                                                    else self.Scaling[1, 1] if image_name == 'RawImageB'
                                                    else 0.0),
                }
            except (ValueError, AttributeError):
                flatten_kwargs_global = None

        fls_flat_by_layer = [self.fls[lid].ravel() for lid in range(nz)]

        # ---- 5. Build & dispatch s0 shards (streamed + staged + vectorized) ---
        # At 100 TB / 820k shards scale, materializing params_s0 as a list would
        # consume ~300+ GB of driver RAM. Instead: yield one shard at a time;
        # the staging loop pulls max_futures-sized batches and submits each.
        n_total_origins = len(s0_origins)

        def _iter_s0_params():
            for origin_out in s0_origins:
                # Convert output-axis-order origin/size to canvas ZYX coords.
                origin_zyx_shard = tuple(origin_out[axis_order.index(a)] for a in 'zyx')
                size_zyx_shard   = tuple(use_s0[axis_order.index(a)]    for a in 'zyx')
                z0, y0, x0 = origin_zyx_shard
                sz, sy, sx = size_zyx_shard
                sz = min(sz, nz - z0); sy = min(sy, ny - y0); sx = min(sx, nx - x0)
                if sz <= 0 or sy <= 0 or sx <= 0:
                    continue

                layer_ids = list(range(z0, z0 + sz))

                # Vectorised overlap test across the z-slab.
                tx_slab  = tile_tx [z0:z0 + sz]
                ty_slab  = tile_ty [z0:z0 + sz]
                tx1_slab = tile_tx1[z0:z0 + sz]
                ty1_slab = tile_ty1[z0:z0 + sz]
                overlap = ~((tx1_slab <= x0) | (tx_slab >= x0 + sx) |
                            (ty1_slab <= y0) | (ty_slab >= y0 + sy))

                # Vectorised contrib gathering — single np.argwhere over the
                # entire (sz, n_tiles) overlap mask, then bucket-sort by layer.
                # Replaces the per-layer Python loop in the original code.
                tiles_per_layer = [[] for _ in range(sz)]
                if overlap.any():
                    li_t = np.argwhere(overlap)                        # (n_contribs, 2)
                    li_arr = li_t[:, 0]
                    t_arr  = li_t[:, 1]
                    order = np.argsort(li_arr, kind='stable')
                    li_arr = li_arr[order]
                    t_arr  = t_arr [order]
                    for li, t in zip(li_arr.tolist(), t_arr.tolist()):
                        lid = z0 + li
                        tiles_per_layer[li].append((
                            t,
                            fls_flat_by_layer[lid][t],
                            tr_matr_all[lid, t],
                            float(tile_I0s_all[lid, t]),
                            float(tile_scales_all[lid, t]),
                        ))

                yield [
                    str(output_zarr_path),
                    int(x0), int(y0), int(z0),
                    int(sx), int(sy), int(sz),
                    layer_ids,
                    tiles_per_layer,
                    image_name, fill_value,
                    weight_min, weight_max,
                    left_crop,
                    flatten_kwargs_global,
                    axis_perm,
                    dtp,
                    U8_range,
                    False,                        # verbose inside worker (driver prints progress)
                ]

        if verbose:
            print('\n' + time.strftime('%Y/%m/%d  %H:%M:%S')
                  + f"   Writing s0:  up to {n_total_origins} shards "
                  + f"(empties skipped), output shape={dst_shape}, dtype={dtp}")

        kwargs_tt = {'interpolation': interpolation,
                     'border_value':  border_value,
                     'border_mode':   border_mode}

        if use_DASK:
            df_future = DASK_client.scatter(deformation_field, broadcast=True)
            s0_iter = _iter_s0_params()
            n_batches_est = (n_total_origins + max_futures - 1) // max_futures
            batch_idx = 0
            pbar_batches = tqdm(total=n_batches_est, desc='Writing s0 (DASK batches)')
            while True:
                batch = list(itertools.islice(s0_iter, max_futures))
                if not batch:
                    break
                batch_idx += 1
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')
                          + '   Submitting s0 batch {:d}: {:d} shards'.format(batch_idx, len(batch)))
                futures = DASK_client.map(
                    _write_zarr3_shard_s0_from_tiles, batch,
                    deformation_field=df_future,
                    retries=DASK_client_retries, **kwargs_tt,
                )
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=f'  s0 shards (batch {batch_idx})',
                                leave=False):
                    fut.result()
                    fut.cancel()
                # Release ~22 GB of contrib lists before building the next batch.
                del batch, futures
                pbar_batches.update(1)
            pbar_batches.close()
        else:
            for p in tqdm(_iter_s0_params(), total=n_total_origins,
                          desc='Writing s0 shards', display=verbose):
                _write_zarr3_shard_s0_from_tiles(p, deformation_field, **kwargs_tt)

        # ---- 6. Build pyramid (s1, s2, ...) ------------------------------
        for lvl in range(1, n_pyramid_levels):
            dst_shape_lvl = level_shapes[lvl]
            use_sh_lvl    = level_shards[lvl]
            origins_lvl = list(itertools.product(
                *[list(range(0, dst_shape_lvl[d], use_sh_lvl[d])) for d in range(3)]
            ))
            params_lvl = []
            for o in origins_lvl:
                dst_slices = tuple(
                    slice(o[d], min(o[d] + use_sh_lvl[d], dst_shape_lvl[d]))
                    for d in range(3)
                )
                src_slices = tuple(
                    slice(dst_slices[d].start * downsample_factor,
                          min(dst_slices[d].stop * downsample_factor, level_shapes[lvl - 1][d]))
                    for d in range(3)
                )
                params_lvl.append([
                    str(output_zarr_path),
                    f's{lvl - 1}', f's{lvl}',
                    src_slices, dst_slices, downsample_factor, dtp,
                ])

            if verbose:
                print('\n' + time.strftime('%Y/%m/%d  %H:%M:%S')
                      + f"   Building s{lvl}:  {len(params_lvl)} shards from s{lvl - 1}")
            if use_DASK:
                n_tasks_lvl   = len(params_lvl)
                n_batches_lvl = (n_tasks_lvl + max_futures - 1) // max_futures
                for batch_idx in tqdm(range(n_batches_lvl),
                                      desc=f'Downsampling to s{lvl} (DASK batches)'):
                    start = batch_idx * max_futures
                    stop  = min(start + max_futures, n_tasks_lvl)
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')
                              + '   Submitting s{:d} batch {:d}/{:d}: shards [{:d}, {:d})'.format(
                                    lvl, batch_idx + 1, n_batches_lvl, start, stop))
                    futures = DASK_client.map(
                        _downsample_zarr3_shard, params_lvl[start:stop],
                        retries=DASK_client_retries,
                    )
                    for fut in tqdm(as_completed(futures), total=len(futures),
                                    desc=f'  s{lvl} shards (batch {batch_idx + 1}/{n_batches_lvl})',
                                    leave=False):
                        fut.result()
                        fut.cancel()
                    del futures
            else:
                for p in tqdm(params_lvl, desc=f'Downsampling to s{lvl}',
                              display=verbose):
                    _downsample_zarr3_shard(p)

        elapsed = time.time() - t_start
        if verbose:
            print('\n' + time.strftime('%Y/%m/%d  %H:%M:%S')
                  + f"   save_stack_zarr3 done in {elapsed/60:.1f} min")

        # ---- 7. Neuroglancer link ----------------------------------------
        ng_link = generate_neuroglancer_link(
            output_zarr_path, layer_name=dataset_name,
            viewer_url=ng_viewer_url, serve_base_url=ng_serve_url,
            display_axes_order=ng_axes_order, zarr_format=3,
        )
        _print_neuroglancer_info(
            output_zarr_path, serve_base_url=ng_serve_url,
            layer_name=dataset_name, viewer_url=ng_viewer_url,
            display_axes_order=ng_axes_order, zarr_format=3,
        )

        return {
            'output_zarr_path': str(output_zarr_path),
            'shape_zyx': src_shape_zyx,
            'shape_out': dst_shape,
            'dtype': str(np.dtype(dtp)),
            'chunks': level_chunks[0],
            'shards': level_shards[0],
            'n_levels': n_pyramid_levels,
            'elapsed_s': elapsed,
            'neuroglancer_link': ng_link,
        }