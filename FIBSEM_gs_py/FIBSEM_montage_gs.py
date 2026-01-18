import numpy as np
import os
import time
import shutil
import psutil
import glob
import pandas as pd
import socket
import platform

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from scipy import sparse
from scipy.sparse.linalg import lsqr
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, warp
from struct import unpack, pack
from tqdm.notebook import tqdm
from collections import defaultdict
import mrcfile
import cv2

from dask.distributed import Client
from dask.distributed import as_completed
from IPython.display import IFrame
from ClusterWrap.clusters import janelia_lsf_cluster

from FIBSEM_gs_py.FIBSEM_gs import (FIBSEM_frame,
                        ShiftTransform,
                        XScaleShiftTransform,
                        ScaleShiftTransform,
                        RegularizedAffineTransform,
                        get_min_max_thresholds,
                        extract_keypoints_descr_files,
                        determine_transformations_files,
                        convert_tr_matr_into_deformation_field,
                        evaluate_FIBSEM_frames_dataset)

from FIBSEM_gs_py.FIBSEM_help_functions_gs import (check_DASK,
                                                    dask_remove_file,
                                                    elapsed_since,
                                                    get_process_memory,
                                                    format_bytes)

def get_adjacent_index_pairs(image_shape):
    """
    Generates an array of index pairs for horizontally and vertically adjacent 
    elements in a 2D image.

    Args:
        image_shape (tuple): A tuple (nrows, ncols) representing the shape of the 2D image.

    Returns:
        numpy.ndarray: An array where each row is a pair of (row1, col1, row2, col2)
                       representing the indices of two adjacent elements.
    """
    nrows, ncols = image_shape

    # Horizontal adjacencies
    h_pairs = []
    for r in range(nrows):
        for c in range(ncols - 1):
            h_pairs.append((r*ncols+c, r*ncols+c+1))
    # Vertical adjacencies
    v_pairs = []
    for r in range(nrows - 1):
        for c in range(ncols):
            v_pairs.append((r*ncols+c, (r+1)*ncols+c))
    # Combine and convert to NumPy array
    all_pairs = np.array(h_pairs + v_pairs)
    return all_pairs

def build_weight_array(shape, **kwargs):
    '''
    Builds a 2D array of weights for image blending. gleb.shtengel@gmail.com 11.2025
    Parameters:
    -----------
    shape : tuple (y, x)
        Shape of the array to create the weights.
    
    kwargs:
    ----------
    weight_min : float
        weight_min for weight. Default is 1
    weight_max : float
        weight_max for weight. Default is 2048
    '''
    weight_min = kwargs.get('weight_min', 1.0)
    weight_max = kwargs.get('weight_max', 1024.0)
    indy, indx = np.indices(shape)
    indx_r = np.flip(indx)
    indy_r = np.flip(indy)
    weights = np.clip((np.min(np.array([indx, indx_r, indy, indy_r]), axis=0) + weight_min), weight_min, weight_max)
    return weights


def transform_tile(tile_params, deformation_field):
    '''
    Transforms individual tile to add to the montage. gleb.shtengel@gmail.com 11.2025
    
    Parameters:
    -----------
    tile_params : list :  j, fl, tr_matr_single, montage_ysz, montage_xsz, weight_min, weight_max
        j : int, tile ID
        fl : str, filename for the tile
        tr_matr_single : 3x3 array : transformation matrix
        montage_xsz : int : montage x-size in pixels
        montage_ysz : int : montage y-size in pixels
        weight_min : float :  weight_min for weight
        weight_max : float :  weight_max for weight

    deformation_field : 3D array
        Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
    
    '''
    j, fl, tr_matr_single, montage_ysz, montage_xsz, weight_min, weight_max, left_crop = tile_params
    fr = FIBSEM_frame(fl)
    tile_initial = fr.RawImageA
    perform_deformation = not np.all(np.isnan(deformation_field))
    if perform_deformation:
        df0 = convert_tr_matr_into_deformation_field(tr_matr_single, (fr.YResolution, fr.XResolution)).astype(np.float32)
        #print(df0.shape, deformation_field.shape)
        df = deformation_field + df0
    else:
        df = convert_tr_matr_into_deformation_field(tr_matr_single, (fr.YResolution, fr.XResolution)).astype(np.float32)
    tile_transformed, shift_x, shift_y = remap_tile(tile_initial, df)
    loc_szy, loc_szx = tile_transformed.shape
    #xi = - shift_x
    #xa = np.min(((xi + loc_szx), montage_xsz-1))
    x0 = np.max((shift_x, 0))
    xi = np.max((- shift_x, 0))
    xa = np.min(((xi + loc_szx), montage_xsz-1))
    #yi = - shift_y
    #ya = np.min(((yi + loc_szy), montage_ysz-1))
    y0 = np.max((shift_y, 0))
    yi = np.max((- shift_y, 0))
    ya = np.min(((yi + loc_szy), montage_ysz-1))
    tile_transformed_cropped = tile_transformed[y0:(ya-yi), x0+left_crop:(xa-xi)]
    weight_out = build_weight_array(tile_transformed_cropped.shape, weight_min = weight_min, weight_max = weight_max)
    weight_out[np.isnan(tile_transformed_cropped)] = 0
    tile_out = np.nan_to_num(tile_transformed_cropped, copy=False, nan=0.0) * weight_out
    return tile_out, weight_out, xi, xa-left_crop-x0, yi,  ya-y0
    

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
    linewidth : float
        Matplotlib linewidth. Default is 1.0.
    color : string
        Matplotlib color. Default 'cyan'.
    tile_positions : 2D array or list
        Actual tile positions. Default is montage_object.tile_positions.
    left_crop : int 
        Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
    dx : int
        X-size of teh tile. Default is montage_object.XResolution-left_crop
    dy : int
        Y-size of teh tile. Default is montage_object.YResolution
    '''
    linestyle = kwargs.get('linestyle', 'dashed')
    linewidth = kwargs.get('linewidth', 1.0)
    color = kwargs.get('color', 'cyan')
    tile_positions_actual = kwargs.get('tile_positions_actual', True)
    if tile_positions_actual:
        tile_positions = kwargs.get('tile_positions', montage_object.tile_positions)

    left_crop = kwargs.get('left_crop', 0)
    
    dx = kwargs.get('dx', montage_object.XResolution-left_crop)
    dy = kwargs.get('dy', montage_object.YResolution)
    
    if tile_positions_actual:
        for tile_position in tile_positions:
            xi, yi = tile_position
            rect_patch = patches.Rectangle((xi,yi), dx-2, dy-2,
            linewidth=linewidth, linestyle=linestyle, edgecolor=color, facecolor='none')
            ax.add_patch(rect_patch)
    else:
        X0 = montage_object.FirstPixels[0, 0]
        Y0 = montage_object.FirstPixels[0, 1]
        for FirstPixel_pair in montage_object.FirstPixels:
            xi = np.max((FirstPixel_pair[0]- X0, 0))
            yi = np.max((FirstPixel_pair[1]- Y0, 0))
            dx_loc = dx  + np.min((FirstPixel_pair[0]- X0, 0))
            dy_loc = dy  + np.min((FirstPixel_pair[1]- Y0, 0))
            rect_patch = patches.Rectangle((xi,yi), dx_loc-2, dy_loc-2,
            linewidth=linewidth, linestyle=linestyle, edgecolor=color, facecolor='none')
            ax.add_patch(rect_patch)


def stiching_bundle_adjustment(number_of_tiles, pairwise_translations, **kwargs):
    '''
    Performs 2D Bundle Adjustment for tile stitching with weighted constraints.
    Builds and solves (in LS sense) the over-constrained system of equations based on pairwise tile translations determined by either SIFT or Cross-Correlation
    gleb.shtengel@gmail.com 12.2025
    
    Parameters:
    -----------
    number_of_tiles : int
        Total number of tiles (0 ... number_of_tiles-1)
    pairwise_translations : dict
        Keys: (i, j) with i < j
        Values: (dx, dy, weight)  OR just (dx, dy)
        If weight is missing, then default_pair_weight is used

    kwargs:
    -----------
    anchors : list of [tile_idx, x, y, weight]  OR [tile_idx, x, y]
        If weight omitted -> default_anchor_weight used
        Default: [[0, 0.0, 0.0, default_pair_weight]] -> fixes the tile 0 at origin.
    default_pair_weight : float, default 1.0
        Weight for pairwise constraints without explicit weight.
    default_anchor_weight : float, default 100.0
        How strongly anchors are enforced (100–10000 typical).
    add_reverse_edges : bool, default False
        If True, adds both (i->j) and (j->i) with same weight (increases robustness).
    damp : float
        Damping coefficient for LSQR solve. Default is 0.
    atol, btol : float, optional
        Stopping tolerances. `lsqr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol. Defaults are atol=1e-10, btol=1e-10.
    iter_lim : int, optional
        Explicit limitation on number of iterations (for safety). Default is 20000.

    Returns:
    -----------
        p_opt : ndarray 
            Array of (x, y) coordinates for each tile. Shape (number_of_tiles, 2).
    '''
    default_pair_weight = kwargs.get('default_pair_weight', 1.0)
    default_anchor_weight = kwargs.get('default_anchor_weight', 100.0)
    add_reverse_edges = kwargs.get('add_reverse_edges', False)
    anchors = kwargs.get('anchors', [[0, 0.0, 0.0, default_anchor_weight]])
    damp = kwargs.get('damp', 0)
    atol = kwargs.get('atol', 1e-10)
    btol = kwargs.get('btol', 1e-10)
    iter_lim = kwargs.get('iter_lim', 20000)
    
    # Normalize the anchor format: ensure every anchor has 4 elements
    anchor_positions = []
    for item in anchors:
        if len(item) == 3:
            tile_idx, x, y = item
            w = default_anchor_weight
        elif len(item) == 4:
            tile_idx, x, y, w = item
        else:
            raise ValueError("anchor_positions items must have 3 or 4 elements: [tile, x, y, (weight)]")
        anchor_positions.append((tile_idx, x, y, w))

    # Build the list of directed edges (weighted) from pairwise translation.
    edges = []
    for (i, j), val in pairwise_translations.items():
        if len(val) == 2:
            dx, dy = val
            w = default_pair_weight
        elif len(val) == 3:
            dx, dy, w = val
        else:
            raise ValueError("pairwise_translations values must be (dx, dy) or (dx, dy, weight)")
        edges.append((i, j, dx, dy, w))
        if add_reverse_edges:
            edges.append((j, i, -dx, -dy, w))

    M = len(edges)  # number of pairwise constraints
    N_anchors = len(anchor_positions)

    # Total equations: 2 per edge (x,y) + 2 per anchor (x,y)
    total_rows = 2 * M + 2 * N_anchors
    total_cols = 2 * number_of_tiles

    row = []
    col = []
    data = []
    b = np.zeros(total_rows)

    # 1. Add pairwise translation constraints (weighted) to the system of equations.
    for eq_idx, (src, dst, dx, dy, weight) in enumerate(edges):
        base = 2 * eq_idx
        w_sqrt = np.sqrt(weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||

        # x-equation: x_src - x_dst = dx  ->  coef [+1 at src, -1 at dst]
        row.extend([base, base])
        col.extend([2 * src, 2 * dst])
        data.extend([w_sqrt, -w_sqrt])
        b[base] = w_sqrt * dx

        # y-equation
        row.extend([base + 1, base + 1])
        col.extend([2 * src + 1, 2 * dst + 1])
        data.extend([w_sqrt, -w_sqrt])
        b[base + 1] = w_sqrt * dy

    # 2. Add anchor constraints (also weighted) to the system of equations.
    anchor_start = 2 * M
    for a_idx, (tile_idx, anc_x, anc_y, weight) in enumerate(anchor_positions):
        base = anchor_start + 2 * a_idx
        w_sqrt = np.sqrt(weight)

        # x_tile = anc_x
        row.append(base)
        col.append(2 * tile_idx)
        data.append(w_sqrt)
        b[base] = w_sqrt * anc_x

        # y_tile = anc_y
        row.append(base + 1)
        col.append(2 * tile_idx + 1)
        data.append(w_sqrt)
        b[base + 1] = w_sqrt * anc_y

    # Build sparse matrix
    A = sparse.csr_matrix((data, (row, col)), shape=(total_rows, total_cols))

    # Solve using LSQR (robust and memory-efficient)
    sol = lsqr(A, b, damp=damp, atol=atol, btol=btol, iter_lim=iter_lim)[0]

    p_opt = sol.reshape(-1, 2)
    return p_opt

def remap_tile(img, deformation_field, **kwargs):
    '''
    Remap Image using CV2.remap (using deformation field). gleb.shtengel@gmail.com 11.2025
    This is needed to work around CV2.remap SHRT_MAX limitation (CV2.remap cannot work with images larger than 32767).
    
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
    
    Returns: image_deformed, shift_x, shift_y
    '''
    interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
    borderValue = kwargs.get('borderValue', np.nan)
    
    img_shape = img.shape
    shift_x = int(np.min((np.nanmin(deformation_field[:, :, 0]), 0)))
    shift_y = int(np.min((np.nanmin(deformation_field[:, :, 1]), 0)))
    df_shifted = deformation_field*1.0
    df_shifted[:, :, 0] = deformation_field[:, :, 0] - shift_x
    df_shifted[:, :, 1] = deformation_field[:, :, 1] - shift_y
    dfx_min = np.nanmin(df_shifted[:, :, 0])
    dfx_max = np.nanmax(df_shifted[:, :, 0])
    dfy_min = np.nanmin(df_shifted[:, :, 1])
    dfy_max = np.nanmax(df_shifted[:, :, 1])
    xsz_new = np.max((img_shape[1], int(dfx_max - dfx_min + 1)))
    ysz_new = np.max((img_shape[0], int(dfy_max - dfy_min + 1)))
    image_expanded = np.zeros((ysz_new, xsz_new))
    image_expanded[0:img_shape[0],0:img_shape[1]] = img
    df_expanded = np.zeros((ysz_new, xsz_new, 2))
    df_expanded[0:img_shape[0],0:img_shape[1], :] = df_shifted
                           
    image_deformed = cv2.remap(image_expanded,
                               df_expanded[:, :, 0].astype(np.float32),
                               df_expanded[:, :, 1].astype(np.float32), interpolation=interpolation, borderValue=borderValue)
    
    return image_deformed, shift_x, shift_y

def find_Transform_ECC(img1, img2, **kwargs):
    '''
    Find Transformation using cv2.findTransformECC on parts of images with approximately known and small image overlaps.
    Works much faster than if performed on whole images. gleb.shtengel@gmail.com 11.2025.
    
    ----------
    Params:
    img1 : 2D array
    img2 : 2D array
    
    ----------
    kwargs:
    image_margins : tuple of 2 ints
        Parts of images to be used. It is assumed that img1 is to the left and above of the img2.
        Subsets img1[-ymargin:, -xmargin:] and  img2[0:ymargin, 0:xmargin] will be used for correlation.
        Default is full images, so image_margins = (ymargin, xmargin) = img1.shape
    warp_matrix : 3x2 initial guess of the transf matrix.
        Default is np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    motion : target transformation.
        Default is cv2.MOTION_TRANSLATION
    criteria : criteria.
        Default is (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    repeats : int
        repeat internally this many times. Default is 2.
    verbose : boolean
            Display intermediate results. Default is False.
            
    Returns: warp_matrix, error_code
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
    If no defonmation is needed, still pass a deformation field, but it will not be use (kwarg['perform_deformation']=False)
    Works much faster than if performed on whole images. gleb.shtengel@gmail.com 11.2025.
    
    ----------
    Params:
    params : list of [fname1, fname2, kwargs]
    deformation_field : 2D array
        Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
    
    ----------
    kwargs:
    interpolation : int
        Interpolation type as defined in CV2. Default is cv2.INTER_LINEAR.
    fill_value = 0.0
        Fill value for outside pixeld in cv2.remap. Default is 0.
    image_margins : tuple of 2 ints
        Parts of images to be used. It is assumed that img1 is to the left and above of the img2.
        Subsets img1[-ymargin:, -xmargin:] and  img2[0:ymargin, 0:xmargin] will be used for correlation.
        Default is full images, so image_margins = (ymargin, xmargin) = img1.shape
    left_crop : int 
        Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
    warp_matrix : 3x2 initial guess of the transf matrix.
        Default is np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    motion : target transformation.
        Default is cv2.MOTION_TRANSLATION
    criteria : criteria.
        Default is (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)
    repeats : int
        repeat internally this many times. Default is 2.
    verbose : boolean
            Display intermediate results. Default is False.
            
    Returns: warp_matrix, error_code
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
        img1 = cv2.remap(FIBSEM_frame(fname1, ftype=ftype).RawImageA_8bit_thresholds()[0].astype(float), deformation_field[:, :, 0].astype(np.float32), deformation_field[:, :, 1].astype(np.float32), interpolation=interpolation, borderValue=fill_value)[:, left_crop:].astype(np.uint8)
        img2 = cv2.remap(FIBSEM_frame(fname2, ftype=ftype).RawImageA_8bit_thresholds()[0].astype(float), deformation_field[:, :, 0].astype(np.float32), deformation_field[:, :, 1].astype(np.float32), interpolation=interpolation, borderValue=fill_value)[:, left_crop:].astype(np.uint8)
    else:
        if verbose:
            print('find_Transform_ECC_DASK: no deformation, left_crop={:d}'.format(left_crop))
        img1 = FIBSEM_frame(fname1, ftype=ftype).RawImageA_8bit_thresholds()[0][:, left_crop:]
        img2 = FIBSEM_frame(fname2, ftype=ftype).RawImageA_8bit_thresholds()[0][:, left_crop:]
    ymargin, xmargin =  kwargs.get('image_margins', img1.shape)
    kwargs['image_margins'] = (ymargin, xmargin-left_crop)
    warp_matrix, error_code = find_Transform_ECC(img1, img2, **kwargs)

    return warp_matrix, error_code


class FIBSEM_montage: 
    '''
    A class representing a FIB-SEM montage - a single z-pane consisting of multiple tiles.
    ©G.Shtengel 10/2025 gleb.shtengel@gmail.com
    Contains the info/settings on the FIB-SEM montage and the procedures that can be performed on it.
    '''
    
    def __init__(self, fls, **kwargs):
        '''
        Initializes (or recalls) an instance of  FIBSEM_montage object. ©G.Shtengel 10/2025 gleb.shtengel@gmail.com

        Parameters:
        ----------
        fls : array of str
            Filenames for the individual data frames in the montage

        kwargs:
        ---------
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif).
        fls_anchors : list/array of str
            Filenames for the individual data frames for anchors.
            Thsese MUST be corresponding files from the previous montage in the same exact order or and empty list (Default).
            In this case the anchoring will be done of the tile 0 to the coordinates x=0.0, y=0.0.
        anchor_input_transformation_matrices : list/array of transformayion matrices
            Should be the same length as fls_anchors. Default is empty list.
        default_pair_weight : float, default 1.0
            Weight for pairwise constraints without explicit weight.
        default_anchor_weight : float, default 100.0
            How strongly anchors are enforced (100–10000 typical).
        add_reverse_edges : bool, default False
            If True, adds both (i->j) and (j->i) with same weight (increases robustness).
        EightBit : int
            If 1 then the data is assumed uint8, otherwise int16
        dump_filename : str
            Filename (full path) to a binary dump file with saved dataset attributes. If dump_filename points to a valid binary file the data set saved in that file will be recalled. Default is empty string ''.
        data_dir : str
            Data directory (path).
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3. Default is 3.
        Sample_ID : str
            Sample ID.
        PixelSize : float
            Pixel size in nm. Default is determined from the frame metadata. If that is not available, default is 8.0.
        Scaling : 2D array of floats
            Scaling parameters allowing to convert I16 data into actual electron counts.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is 1e-3.
        thr_max : float
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
            The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. SIFT library default is 0.04.
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
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            Matrix of regularization (shrinkage) parameters (applicable only if RegularizedAffineTransform is used). Default is 1e-5.
        targ_vector = 1D float array
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
            Min number of matches for the transformation to be considered valid. Delault is 5.
        int_order : int
            The order of interpolation (when transforming the data).
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        dtp : Data Type
            Python data type for saving. Default is np.int16.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        perform_deformation : boolean
            If True - the data is deformed (in addition to tyransformation defined above) using the deformation field data defined below.
        deformation_type : str
            Type of Deformation. Options are:
                'post_1DY'  - Default. Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
                'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
        deformation_sigma :  list of 1 or two floats.
            Gaussian width of smoothing (units of pixels). Default is 50.    
        disp_res : boolean
            If False, the intermediate printouts will be suppressed. Default is True.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is True.
        '''  
        disp_res = kwargs.get('disp_res', True)
        self.fls = fls
        self.data_dir = kwargs.get('data_dir', os.path.split(fls[0])[0])
        self.ftype = kwargs.get('ftype', 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        self.fls_anchors = kwargs.get('fls_anchors', [])
        self.anchor_input_transformation_matrices = kwargs.get('anchor_input_transformation_matrices', [np.eye(3,3) for f in self.fls_anchors])
        if (len(self.fls_anchors) != len(self.anchor_input_transformation_matrices)):
            raise ValueError('fls_anchors and anchor_input_transformation_matrices should have same number of elements')
        self.default_pair_weight = kwargs.get('default_pair_weight', 1.0)
        self.default_anchor_weight = kwargs.get('default_anchor_weight', 100.0)
        self.add_reverse_edges = kwargs.get('add_reverse_edges', False)
        test_frame = FIBSEM_frame(fls[0], ftype = self.ftype, calculate_scaled_images=False, read_header_only=True)
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
        self.XResolutions = kwargs.get('XResolutions', np.full(len(fls), test_frame.XResolution))
        self.YResolutions = kwargs.get('YResolutions', np.full(len(fls), test_frame.YResolution))
        self.Scaling = kwargs.get("Scaling", test_frame.Scaling)
        if hasattr(test_frame, 'PixelSize'):
            self.PixelSize = kwargs.get("PixelSize", test_frame.PixelSize)
        else:
            self.PixelSize = kwargs.get("PixelSize", 8.0)
        self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

        self.DetA = test_frame.DetA
        self.DetB = test_frame.DetB
        self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        self.Sample_ID = kwargs.get("Sample_ID", test_frame.Sample_ID)
        self.EightBit = kwargs.get("EightBit", 1)
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
        self.max_iter = kwargs.get("max_iter", 1000)
        self.SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
        self.save_res_png  = kwargs.get("save_res_png", True)
        self.fnm_types = kwargs.get("fnm_types", ['mrc'])
        self.flipY = kwargs.get("flipY", False)                     # If True, the registered data will be flipped along Y axis
                                                                    # window size 701, polynomial order 3
        self.int_order = kwargs.get("int_order", False)             #     The order of interpolation. The order has to be in the range 0-5:
                                                                    #    - 0: Nearest-neighbor
                                                                    #    - 1: Bi-linear (default)
                                                                    #    - 2: Bi-quadratic
                                                                    #    - 3: Bi-cubic
                                                                    #    - 4: Bi-quartic
                                                                    #    - 5: Bi-quintic
        
        self.pad_edges =  kwargs.get("pad_edges", True)
        self.perform_deformation = kwargs.get("perform_deformation", False)
        self.deformation_type = kwargs.get("deformation_type", 'post_1DY')
        self.deformation_sigma = kwargs.get('deformation_sigma', 50)
        try:
            build_fnm_montage = os.path.splitext(os.path.split(fls[0])[1])[0][0:-5] + 'montage.dat'
        except:
            build_fnm_montage = 'montage.dat'
        self.fnm_montage = kwargs.get("fnm_montage", build_fnm_montage)
        self.dtp = kwargs.get("dtp", np.int16)
        kwargs.update({'data_dir' : self.data_dir, 'fnm_montage' : self.fnm_montage, 'dtp' : self.dtp})
               
        FirstPixels = []
        for fl in fls:
            fr = FIBSEM_frame(fl, read_header_only=True)
            FirstPixels.append([fr.FirstPixelX, fr.FirstPixelY])
        self.FirstPixels = np.array(FirstPixels)
        
        if 'shape' in kwargs:
            self.shape = kwargs['shape']
        else:
            # try to auto-determine shape and adjacent pairs
            try:
                tile_string = os.path.splitext(os.path.split(fls[-1])[1])[0][-5:].split('-')
                nrows = int(tile_string[1])+1
                ncols = int(tile_string[2])+1
                self.shape = (nrows, ncols)
            except:
                print('Could not auto-determine the shape, and therefore the montage size and the adjacent tile pairs')
                print('Define the montage size (self.Xsize, self.Ysize) manually')
                print('Define the adjacent tile pairs (self.adjacent_pairs - list of indices of files of the adjacent tiles) manually')
                self.shape = (1, 1)
        
        if self.shape[1] > 1:
            self.Xoverlap = self.XResolution - (self.FirstPixels[1, 0] - self.FirstPixels[0, 0])
        else:
            self.Xoverlap = 0
        if self.shape[0] > 1:
            self.Yoverlap = self.YResolution - (self.FirstPixels[self.shape[1], 1] - self.FirstPixels[(self.shape[1]-1), 1])
        else:
            self.Yoverlap = 0
        self.fnms_kpts = ['' for fl in fls]
        
        # these are for pairwise translations (transformations) within a single montage mosaic
        self.adjacent_pairs = get_adjacent_index_pairs(self.shape)
        self.SIFT_transformation_valid = [False for m in self.adjacent_pairs]
        self.ECC_transformation_valid = [False for m in self.adjacent_pairs]
        self.SIFT_transformation_matrices = [np.eye(3,3) for m in self.adjacent_pairs]
        self.ECC_transformation_matrices = [np.eye(3,3) for m in self.adjacent_pairs]
        self.npts = [0 for m in self.adjacent_pairs ]
        self.fnms_matches = ['' for m in self.adjacent_pairs ]
        
        # these are for translations (transformations) relative to anchor tiles (from previous z-slice) 
        self.ECC_anchor_transformation_matrices = [np.eye(3,3) for m in self.fls]
        self.ECC_anchor_transformation_valid = [False for m in self.fls]
        self.SIFT_anchor_transformation_matrices = [np.eye(3,3) for m in self.fls]
        self.SIFT_anchor_transformation_valid = [False for m in self.fls]
        
        # initialize the montage size (assuming rectangular shape)
        self.Xsize = self.shape[1] * (self.XResolution - self.Xoverlap) + self.Xoverlap
        self.Ysize = self.shape[0] * (self.YResolution - self.Yoverlap) + self.Yoverlap
        
        # initialize the translation matrix for each tile
        shifts_x = self.FirstPixels[:, 0] - self.FirstPixels[0, 0]
        shifts_y = self.FirstPixels[:, 1] - self.FirstPixels[0, 1]
        self.tr_matr = np.tile(np.eye(3,3), (np.product(self.shape), 1, 1))
        self.tr_matr[:, 0, 2] = - np.array(shifts_x).flatten()
        self.tr_matr[:, 1, 2] = - np.array(shifts_y).flatten()
        self.valid_transformations = np.zeros(np.product(self.shape), dtype=int)
        
        # initialize valid translations as "invalid"
        
    def evaluate_FIBSEM_statistics(self, **kwargs):
        '''
        Evaluates parameters of FIBSEM montage (Min/Max, Working Distance (WD), Milling Y Voltage (MV), FOV center positions). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        frame_inds : array
            Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames.
        data_dir : str
            Data directory (path). Default is object attribute.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        FIBSEM_Data_xlsx : str
            File path of the Excell file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).
        use_existing_data : boolean
            Default is False. If True and the data exists (saved into XLSX), use that.            
        verbose : boolean
            If True, intermediate messages and results will be displayed. Default is False.

        Returns:
        list of 14 parameters: FIBSEM_Data_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions
            FIBSEM_Data_xlsx : str
                path to Excel file with the FIBSEM data
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8)
            data_man_glob : float   
                max data value for I8 conversion (open CV SIFT requires I8)
            center_x : float array
                FOV Center X-coordinate extracted from the header data
            center_y : float array
                FOV Center Y-coordinate extracted from the header data
            ScanRate : float array
                SEM Scan Rate (Hz)
            EHT : float array
                SEM EHT voltage (kV)
            SEMSpecimenI : float array
                SEM Specimen current (nA)
            XResolutions : int array
                X-frame sizes
            YResolutions : int array
                Y-frame sizes
        '''
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        frame_inds = kwargs.get("frame_inds", np.arange(len(self.fls)))
        data_dir = kwargs.get('data_dir', self.data_dir)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        fit_params = kwargs.get('fit_params', ['SG', 3, 1])

        FIBSEM_Data_xlsx_default = os.path.join(data_dir, self.fnm_montage.replace('.dat', '_FIBSEM_Data.xlsx'))
        FIBSEM_Data_xlsx = kwargs.get('FIBSEM_Data_xlsx', FIBSEM_Data_xlsx_default)
        verbose = kwargs.get('verbose', False)
        use_existing_data = kwargs.get('use_existing_data', False)

        local_kwargs = {'use_DASK' : use_DASK,
                        'DASK_client_retries' : DASK_client_retries,
                        'ftype' : ftype,
                        'frame_inds' : frame_inds,
                        'data_dir' : data_dir,
                        'thr_min' : thr_min,
                        'thr_max' : thr_max,
                        'nbins' : nbins,
                        'sliding_minmax' : False,
                        'fit_params' : fit_params,
                        'FIBSEM_Data_xlsx' : FIBSEM_Data_xlsx,
                        'verbose' : verbose,
                        'use_existing_data' : use_existing_data}

        if verbose:
            print('Evaluating the parameters of FIBSEM data set (data Min/Max, Working Distance, FOV center positions, Scan Rate, EHT)')
        self.FIBSEM_Data = evaluate_FIBSEM_frames_dataset(self.fls, DASK_client, **local_kwargs)
        self.data_minmax = self.FIBSEM_Data[0:5]
        self.data_min_glob = self.FIBSEM_Data[1]
        self.data_max_glob = self.FIBSEM_Data[2]
        
        self.FOV_x = self.FIBSEM_Data[7]
        self.FOV_y = self.FIBSEM_Data[8]
        try:
            self.XResolutions = self.FIBSEM_Data[12].astype(int)
            self.YResolutions = self.FIBSEM_Data[13].astype(int)
        except:
            self.XResolutions = np.full(len(WD), self.XResolution).astype(int)
            self.YResolutions = np.full(len(WD), self.YResolution).astype(int)

        self.XResolution = np.max(self.XResolutions)
        self.YResolution = np.max(self.YResolutions)

        return self.FIBSEM_Data
    
    def extract_keypoints(self, **kwargs):
        '''
        Extract Key-Points and Descriptors. ©G.Shtengel 10/2025 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. DASK client. If empty string '' (Default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
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
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        data_minmax : list of 5 parameters
            minmax_xlsx : str
                path to Excel file with Min/Max data.
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8).
            data_min_sliding : float array
                min data values (one per file) for I8 conversion.
            data_max_sliding : float array
                max data values (one per file) for I8 conversion.
            data_minmax_glob : 2D float array
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
        deformation_field : 2D array
             Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
        deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        interpolation : int
            Interpolation type as defined in CV2 (if deformation_field is not np.nan) . Default is cv2.INTER_LINEAR.
        fill_value = 0.0
            Fill value for outside pixeld in cv2.remap. Default is 0.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
        If True, outputs will be printed.
    
        Returns:
        fnms_kpts : array of str
            Filenames for binary files containing Key-Points and Descriptors for each frame.
        '''
        verbose = kwargs.get('verbose', True)
        if len(self.fls) == 0:
            if verbose:
                print('Data set not defined, perform initialization first')
            fnms_kpts = []
        else:  
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client, verbose = verbose)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            ftype = kwargs.get("ftype", self.ftype)
            thr_min = kwargs.get("thr_min", self.thr_min)
            thr_max = kwargs.get("thr_max", self.thr_max)
            nbins = kwargs.get("nbins", self.nbins)
            data_minmax = kwargs.get("data_minmax", self.data_minmax)
            SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
            SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
            SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
            SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
            SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
            deformation_field = kwargs.get('deformation_field', np.nan)
            perform_deformation = not np.any(np.isnan(deformation_field))
            interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
            fill_value = kwargs.get('fill_value', 0)
            use_existing_data = kwargs.get('use_existing_data', False)

            minmax_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
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
                        'fill_value' : fill_value}

            params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls]        
            if use_DASK:
                shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                futures_s3 = DASK_client.map(extract_keypoints_descr_files, params_s3, deformation_field = shared_data_future, retries = DASK_client_retries)
                fnms_kpts = DASK_client.gather(futures_s3)
            else:
                fnms_kpts = []
                for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ', display=verbose)):
                    fnms_kpts.append(extract_keypoints_descr_files(param_s3, deformation_field))
            self.fnms_kpts = fnms_kpts
        return fnms_kpts
    
    def determine_transformations_SIFT(self, **kwargs):
        '''
        Determine transformation matrices for frame pairs using SIFT. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        target_pairs : pairs of tiles for which the transormations are to be determined. Default is object attribute self.adjacent_pairs (all adjacent pairs).
        TransformType : object reference
            Transformation model used for determining the transformation matrix from Key-Point pairs. Default is object attribute.
            Choose from the following options:
                ShiftTransform - only x-shift and y-shift
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
           Matrix of regularization (shrinkage) parameters (applicable only if RegularizedAffineTransform is used). Default is object attribute.
        targ_vector = 1D float array
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
            Min number of matches for the transformation to be considered valid. Delault is 5.
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
        transformations_results : array of lists containing the results:
            [transformation_matrix, fnm_matches, npt, error_abs_mean, error_FWHMx, error_FWHMy, iteration]
            transformation_matrix : 2D float array
                Transformation matrix for each sequential frame pair.
            fnm_matches : str
                Filename containing the matches used to determine the transformation for the pair of frames.
            npts : int
                Number of matches.
            error_abs_mean : float
                Mean abs error of registration for all matched Key-Points.
        '''
        verbose = kwargs.get('verbose', False)
        if len(self.fnms_kpts) == 0:
            if verbose:
                print('No data on individual key-point data files, peform key-point search')
            transformations_results = []
        else:
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client, verbose = verbose)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            ftype = kwargs.get("ftype", self.ftype)
            target_pairs = kwargs.get('target_pairs', self.adjacent_pairs)
            TransformType = kwargs.get("TransformType", self.TransformType)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            if hasattr('self', 'SIFT_nmatches_min'):
                SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
            else:
                SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
            Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            save_res_png  = kwargs.get("save_res_png", self.save_res_png )
            start = kwargs.get('start', 'edges')
            estimation = kwargs.get('estimation', 'interval')
            use_existing_data = kwargs.get('use_existing_data', False)
            dt_kwargs = {'ftype' : ftype,
                            'TransformType' : TransformType,
                            'l2_matrix' : l2_matrix,
                            'targ_vector': targ_vector, 
                            'solver' : solver,
                            'RANSAC_initial_fraction' : RANSAC_initial_fraction,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'fnm_matches' : fnms_matches,
                            'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
                            'start' : start,
                            'estimation' : estimation,
                            'use_existing_data' : use_existing_data}
            params_SIFT = []
            for j, index_pair in enumerate(target_pairs):
                fname1 = self.fnms_kpts[index_pair[0]]
                fname2 = self.fnms_kpts[index_pair[1]]
                path_base, f1 = os.path.split(fname1)
                _, f2 = os.path.split(fname2)
                fnm_matches = os.path.join(path_base, f1.replace('_kpdes.bin', '_')+f2.replace('_kpdes.bin', '_matches.bin'))
                dt_kwargs['fnm_matches'] = fnm_matches
                params_SIFT.append([fname1, fname2, dt_kwargs])
                if verbose:
                    print('Added a set: ')
                    print([fname1, fname2, dt_kwargs])
            if use_DASK:
                futures_SIFT = DASK_client.map(determine_transformations_files, params_SIFT, retries = DASK_client_retries)                
                transformations_results = DASK_client.gather(futures_SIFT)
            else:
                transformations_results = []
                for param_SIFT in tqdm(params_SIFT, desc = 'Extracting Transformation Parameters: ', display=verbose):
                    transformations_results.append(determine_transformations_files(param_SIFT))
            
            for (j, index_pair), transformations_result in zip(enumerate(target_pairs), transformations_results):
                try:
                    ind = np.where((self.adjacent_pairs == index_pair).all(axis=1))[0][0]
                    self.SIFT_transformation_matrices[ind] = np.nan_to_num(transformations_result[0])
                    self.fnms_matches[ind] = transformations_result[1]
                    self.npts[ind] = len(transformations_result[2][0])
                    self.SIFT_transformation_valid[ind] = self.npts[ind] > SIFT_nmatches_min
                except Exception as e:
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   An error occurred: {}'.format(e))
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Target pair not in the self.adjacent_pairs list: ', index_pair)              
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints :', np.mean(self.npts).astype(np.int64))
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Valid SIFT transformation established: ', self.SIFT_transformation_valid)
        return transformations_results
    
    def determine_transformations_ECC(self, **kwargs):
        '''
        Determine transformation matrices for frame pairs using ECC. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com
        Uses find_Transform_ECC(img1, img2, **kwargs).
        
        kwargs:
        ---------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        target_pairs : pairs of tiles for which the transormations are to be determined.
            Default is object attribute self.adjacent_pairs (all adjacent pairs).
        image_margins : array or list of tuples of 2 ints
            Parts of images to be used. It is assumed that first image (img1) in each target_pair is to the left and above of the second image (img2).
            Subsets img1[-ymargin:, :] and  img2[0:ymargin, :] or img1[:, -xmargin:] and  img2[:, 0:xmargin] will be used for correlation.
            Default is full images, so image_margins = (self.YResolution, self.XResolution)
        image_margin_factors : tuple of 2 floats
            Multipliers for default image margins. Default is (1.0, 1.0)
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        motion : target transformation.
            Default is cv2.MOTION_TRANSLATION
        repeats : int
            repeat internally this many times. Default is 2.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
            Display intermediate results. Default is True.
        
        Returns:
        transformations_results : array of lists containing the results:
            [transformation_matrix, error_code]
            transformation_matrix : 2D float array
                Transformation matrix for each sequential frame pair.
            error_code : int
                CV2 error code.
        '''
        ftype = kwargs.get('ftype', self.ftype)
        target_pairs = kwargs.get('target_pairs', self.adjacent_pairs)
        repeats = kwargs.get('repeats', 2)
        verbose = kwargs.get('verbose', False)
        use_existing_data = kwargs.get('use_existing_data', False)
        image_margin_factors = kwargs.get('image_margin_factors', (1.0, 1.0))
        deformation_field = kwargs.get('deformation_field', np.nan)
        left_crop = kwargs.get('left_crop', 0)

        if hasattr(self, 'motion'):
            motion = kwargs.get('motion', self.motion)
        else:
            motion = kwargs.get('motion', cv2.MOTION_TRANSLATION)
        if hasattr(self, 'criteria'):
            criteria = kwargs.get('criteria', self.criteria)
        else:
            criteria = kwargs.get('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7))
        
        if len(target_pairs) == 0:
            if verbose:
                print('No target pairs selected')
            transformations_results = []
        else:
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client, verbose = verbose)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            use_existing_data = kwargs.get('use_existing_data', False)
            params_ECC = []
            image_margins_new = []
            for j, index_pair in enumerate(target_pairs):
                dt_kwargs = {'ftype' : ftype,
                         'motion' : motion,
                         'criteria' : criteria,
                         'use_existing_data' : use_existing_data,
                         'verbose' : verbose}
                fname1 = self.fls[index_pair[0]]
                fname2 = self.fls[index_pair[1]]
                FirstPixels_delta = self.FirstPixels[index_pair[1]] - self.FirstPixels[index_pair[0]]
                if 'image_margins' in kwargs:
                    ymargin, xmargin = kwargs['image_margins'][j]
                else:
                    if hasattr(self, 'image_margins'):
                        ymargin, xmargin = self.image_margins[j]
                    else:
                        ymargin = np.min((int(image_margin_factors[0] * (self.YResolution - FirstPixels_delta[1])), self.YResolution))
                        xmargin = np.min((int(image_margin_factors[1] * (self.XResolution - FirstPixels_delta[0])), self.XResolution))
                dt_kwargs['warp_matrix'] = np.array([[1, 0, -FirstPixels_delta[0]], [0, 1, -FirstPixels_delta[1]]], dtype=np.float32)
                dt_kwargs['image_margins'] = (ymargin, xmargin)
                dt_kwargs['left_crop'] = left_crop
                image_margins_new.append([ymargin, xmargin])   
                param_ECC = [fname1, fname2, dt_kwargs]
                params_ECC.append(param_ECC)
            #self.image_margins = np.array(image_margins_new)
            if use_DASK:
                shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                futures_ECC = DASK_client.map(find_Transform_ECC_DASK, params_ECC, deformation_field = shared_data_future, retries = DASK_client_retries)
                transformations_results = DASK_client.gather(futures_ECC)
            else:
                transformations_results = []
                for param_ECC in tqdm(params_ECC, desc = 'Extracting transformation parameters: ', display=verbose):
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Determining transformation params for:')
                        print(param_ECC)
                    transformations_results.append(find_Transform_ECC_DASK(param_ECC, deformation_field))
            #determine_transformations_files returns (transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration)
            for (j, index_pair), transformations_result in zip(enumerate(target_pairs), transformations_results):
                try:
                    ind = np.where((self.adjacent_pairs == np.array(index_pair)).all(axis=1))[0][0]
                    self.ECC_transformation_matrices[ind] = np.nan_to_num(transformations_result[0])
                    self.ECC_transformation_valid[ind] = transformations_result[1] == 0
                except Exception as e:
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   An error occurred: {}'.format(e))
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Target pair not in the self.adjacent_pairs list: ', index_pair)
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Valid ECC transformation established: ', self.ECC_transformation_valid)
        return transformations_results
    
    def determine_anchor_transformations_ECC(self, **kwargs):
        '''
        Determine transformation matrices for frames relative to anchor frames using ECC. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com
        Uses find_Transform_ECC(img1, img2, **kwargs).
        
        kwargs:
        ---------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        target_tiles : target tiles for which the transormations are to be determined.
            Default is object attribute self.fls_anchors.
        image_margins : tuple of 2 ints
            Parts of images to be used. It is assumed that first image (img1) in each target_pair is to the left and above of the second image (img2).
            Subsets img1[-ymargin:, :] and  img2[0:ymargin, :] or img1[:, -xmargin:] and  img2[:, 0:xmargin] will be used for correlation.
            Default is full images, so image_margins = (self.YResolution, self.XResolution)
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        motion : target transformation.
            Default is cv2.MOTION_TRANSLATION
        repeats : int
            repeat internally this many times. Default is 2.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
            Display intermediate results. Default is True.
        
        Returns:
        transformations_results : array of lists containing the results:
            [transformation_matrix, error_code]
            transformation_matrix : 2D float array
                Transformation matrix for each sequential frame pair.
            error_code : int
                CV2 error code.
        '''
        ftype = kwargs.get('ftype', self.ftype)
        target_tiles = kwargs.get('target_tiles', np.arange(len(self.fls_anchors)))
        repeats = kwargs.get('repeats', 2)
        verbose = kwargs.get('verbose', False)
        use_existing_data = kwargs.get('use_existing_data', False)
        image_margins = kwargs.get('image_margins', (self.YResolution, self.XResolution))
        deformation_field = kwargs.get('deformation_field', np.nan)
        left_crop = kwargs.get('left_crop', 0)
        if hasattr(self, 'motion'):
            motion = kwargs.get('motion', self.motion)
        else:
            motion = kwargs.get('motion', cv2.MOTION_TRANSLATION)
        if hasattr(self, 'criteria'):
            criteria = kwargs.get('criteria', self.criteria)
        else:
            criteria = kwargs.get('criteria', (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7))
        
        if len(np.array(self.fls_anchors)[target_tiles]) == 0:
            if verbose:
                print('No target tiles selected or no anchors files selected')
            transformations_results = []
        else:
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client, verbose = verbose)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            use_existing_data = kwargs.get('use_existing_data', False)
            params_ECC = []
            for target_tile in target_tiles:
                dt_kwargs = {'ftype' : ftype,
                         'motion' : motion,
                         'criteria' : criteria,
                         'use_existing_data' : use_existing_data,
                         'verbose' : verbose}
                fname2 = self.fls[target_tile]
                fname1 = self.fls_anchors[target_tile]
                #fname1 = self.fls[target_tile]
                #fname2 = self.fls_anchors[target_tile]
                dt_kwargs['warp_matrix'] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                dt_kwargs['image_margins'] = image_margins
                dt_kwargs['left_crop'] = left_crop
                param_ECC = [fname1, fname2, dt_kwargs]
                params_ECC.append(param_ECC)
            if use_DASK:
                shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                futures_ECC = DASK_client.map(find_Transform_ECC_DASK, params_ECC, deformation_field = shared_data_future, retries = DASK_client_retries)
                transformations_results = DASK_client.gather(futures_ECC)
            else:
                transformations_results = []
                for param_ECC in tqdm(params_ECC, desc = 'Extracting anchor transformation parameters: ', display=verbose):
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Determining anchor transformation params for:')
                        print(param_ECC)
                    transformations_results.append(find_Transform_ECC_DASK(param_ECC, deformation_field))
            for target_tile, transformations_result in zip(target_tiles, transformations_results):
                try:
                    self.ECC_anchor_transformation_matrices[target_tile] = np.nan_to_num(transformations_result[0])
                    self.ECC_anchor_transformation_valid[target_tile] = transformations_result[1] == 0
                except Exception as e:
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   An error occurred: {}'.format(e))
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Target pair not in the self.adjacent_pairs list: ', index_pair)
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Valid ECC anchor transformation established: ', self.ECC_transformation_valid)
        return transformations_results
    
    def solve_montage_stitching(self, **kwargs):
        '''
        Perform Bundle Optimization to get optimal montage stiching. gleb.shtengel@gmail.com 10.2025
        
        kwargs:
        ----------
        anchors : list of [tile_idx, x, y, weight]  OR [tile_idx, x, y]
            If weight omitted -> default_anchor_weight used
            Default: [[0, 0.0, 0.0, default_pair_weight]] -> fixes the tile 0 at origin.
        default_pair_weight : float, default 1.0
            Weight for pairwise constraints without explicit weight.
        default_anchor_weight : float, default 100.0
            How strongly anchors are enforced (100–10000 typical).
        add_reverse_edges : bool, default False
            If True, adds both (i->j) and (j->i) with same weight (increases robustness).    
        verbose : boolean
            Display intermediate results. Default is False.
            
        Returns:
        tile_positions : ndarray 
            Array of (x, y) coordinates for each tile. Shape is (number_of_tiles, 2).
        '''
        if hasattr(self, 'default_pair_weight'):
            default_pair_weight = kwargs.get('default_pair_weight', self.default_pair_weight)
        else:
            default_pair_weight = kwargs.get('default_pair_weight', 1.0)
        if hasattr(self, 'default_anchor_weight'):
            default_anchor_weight = kwargs.get('default_anchor_weight', self.default_anchor_weight)
        else:
            default_anchor_weight = kwargs.get('default_anchor_weight', 100.0)
        if hasattr(self, 'add_reverse_edges'):
            add_reverse_edges = kwargs.get('add_reverse_edges', self.add_reverse_edges)
        else:
            add_reverse_edges = kwargs.get('add_reverse_edges', False)
        anchors = kwargs.get('anchors', [[0, 0.0, 0.0, default_anchor_weight]])
        valid_anchors = [anchor[0] for anchor in anchors]
        verbose = kwargs.get('verbose', False)
        kwargs['default_pair_weight'] = default_pair_weight
        kwargs['default_anchor_weight'] = default_anchor_weight
        kwargs['add_reverse_edges'] = add_reverse_edges
        kwargs['anchors'] = anchors
        kwargs['verbose'] = verbose
        if verbose:
            print('solve_montage_stitching : using following kwargs:')
            print(kwargs)
        adjacent_pairs_loc = []
        tr_matrs_loc = []
        pairwise_trans = {}
        for j, adjacent_pair in enumerate(self.adjacent_pairs):
            if self.SIFT_transformation_valid[j]:
                adjacent_pairs_loc.append(adjacent_pair)
                tr_matrs_loc.append(self.SIFT_transformation_matrices[j])
            else:
                if self.ECC_transformation_valid[j]:
                    adjacent_pairs_loc.append(adjacent_pair)
                    tr_matrs_loc.append(self.ECC_transformation_matrices[j])
                    
        for adjacent_pair, tr_matr in zip(adjacent_pairs_loc, tr_matrs_loc):
            pairwise_trans[(adjacent_pair[0], adjacent_pair[1])] = (tr_matr[0, 2], tr_matr[1, 2])
        
        N = np.product(self.shape)
            
        tile_positions = stiching_bundle_adjustment(N, pairwise_trans, **kwargs)
        
        self.tile_positions = tile_positions
        return tile_positions
    
    def update_translation_matrices(self, **kwargs):
        '''
        Update transformation matrices for each tile based on positions data determined by solve_montage_stitching.
        
        kwargs:
        ----------
        tile_positions : list of pairs of floats.
            Positions to be used. Default is object attribute.
        tolerances : list of two ints
            tolerances for tile position update. If the new position differs from the init position by more than tolerance, the position is not updated.
        verbose : boolean
            Display intermediate results. Default is False.
            
        Returns:
        ----------
        tr_matr_new : ndarray 
            Array of transformation matrices for each tile. Shape is (number_of_tiles, 2, 3).
        '''
        verbose = kwargs.get('verbose', False)
        
        if hasattr(self, 'tile_positions'):
            tile_positions = kwargs.get('tile_positions', self.tile_positions)
        else:
            tile_positions = kwargs.get('tile_positions', [])
        tolerances = kwargs.get('tolerances', [self.XResolution//2, self.YResolution//2])
        if verbose:
            print('Using Tolerances:  Xtol = {:d} (pix),   Ytol = {:d} (pix)'.format(tolerances[0], tolerances[1]))
        tr_matr_new = self.tr_matr * 1.0
        if len(tile_positions)>0:
            for j, pos in enumerate(tile_positions):
                if verbose:
                    print('Tile: {:d}.  Initial X-Position = {:.2f},  New X-Position = {:.2f}'.format(j, -tr_matr_new[j, 0, 2], pos[0]))
                    print('Tile: {:d}.  Initial Y-Position = {:.2f},  New Y-Position = {:.2f}'.format(j, -tr_matr_new[j, 1, 2], pos[1]))
                if np.abs((self.tr_matr[j, 0, 2] + pos[0]))<tolerances[0] and np.abs((self.tr_matr[j, 1, 2] + pos[1]))<tolerances[1]:
                    tr_matr_new[j, 0, 2] = -pos[0]
                    tr_matr_new[j, 1, 2] = -pos[1]
                    self.valid_transformations[j] = 1
                    if verbose:
                        print('Position updated as valid')
                else:
                    if verbose:
                        print('Position NOT updated (New position invalid)')
            self.tr_matr = tr_matr_new
        return tr_matr_new
    
    def register_montage(self, **kwargs):
        '''
        Register the montage - determine the transformation matrices for each tile.
        
        kwargs:
        ----------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        verbose : boolean
            Display intermediate results. Default is False.            
        method : string
            Options are: ['SIFT-ECC', 'SIFT', 'ECC']. Default is 'SIFT-ECC' - try SIFT first, and for the tiles that SIFT failed, try ECC.
        SIFT kwargs:
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        anchors : list of [tile_idx, x, y, weight]  OR [tile_idx, x, y]
            If weight omitted -> default_anchor_weight used
            Default: [[0, 0.0, 0.0, default_pair_weight]] -> fixes the tile 0 at origin.
        default_pair_weight : float, default 1.0
            Weight for pairwise constraints without explicit weight.
        default_anchor_weight : float, default 100.0
            How strongly anchors are enforced (100–10000 typical).
        add_reverse_edges : bool, default False
            If True, adds both (i->j) and (j->i) with same weight (increases robustness). 
        tolerances : list of two ints
            tolerances for tile position update. If the new position differs from the init position by more than tolerance, the position is not updated.

        Returns:
        ----------
        tr_matr_new : ndarray 
            Array of transformation matrices for each tile. Shape is (number_of_tiles, 2, 3).
        
        '''
        use_existing_data = kwargs.get('use_existing_data', False)
        kwargs['use_existing_data'] = use_existing_data
        verbose = kwargs.get('verbose', False)
        kwargs['verbose'] = verbose
        DASK_client = kwargs.get('DASK_client', '')
        kwargs['DASK_client'] = DASK_client
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        kwargs['DASK_client_retries'] = DASK_client_retries
        target_pairs = self.adjacent_pairs
        if hasattr(self, 'SIFT_nmatches_min'):
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
        else:
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min',  5)
        kwargs['SIFT_nmatches_min'] = SIFT_nmatches_min       
        if hasattr(self, 'default_pair_weight'):
            default_pair_weight = kwargs.get('default_pair_weight', self.default_pair_weight)
        else:
            default_pair_weight = kwargs.get('default_pair_weight', 1.0)
        if hasattr(self, 'default_anchor_weight'):
            default_anchor_weight = kwargs.get('default_anchor_weight', self.default_anchor_weight)
        else:
            default_anchor_weight = kwargs.get('default_anchor_weight', 100.0)
        if hasattr(self, 'add_reverse_edges'):
            add_reverse_edges = kwargs.get('add_reverse_edges', self.add_reverse_edges)
        else:
            add_reverse_edges = kwargs.get('add_reverse_edges', False)
        anchors = kwargs.get('anchors', [[0, 0.0, 0.0, default_anchor_weight]])
        kwargs['default_pair_weight'] = default_pair_weight
        kwargs['default_anchor_weight'] = default_anchor_weight
        kwargs['add_reverse_edges'] = add_reverse_edges
        kwargs['anchors'] = anchors
        method = kwargs.get('method', 'SIFT-ECC')
        valid_methods = ['SIFT-ECC', 'SIFT', 'ECC']
        if method not in valid_methods:
            if verbose:
                print('Method ' + method +' is not among valid methods: ', valid_methods)
            return np.nan
        else:
            if method == 'SIFT-ECC' or method == 'SIFT':
                if not hasattr(self, 'FIBSEM_Data'):
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'  Evaluating FIBSEM statistics')
                    FIBSEM_data = self.evaluate_FIBSEM_statistics(fit_params=['SG', 3, 1], **kwargs)
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'  Determining adjacent tile transformations using SIFT')
                fnms = self.extract_keypoints(**kwargs)
                transformations_results = self.determine_transformations_SIFT(**kwargs)
                if method == 'SIFT-ECC':
                    target_pairs = self.adjacent_pairs[np.array(self.SIFT_transformation_valid)==False]
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'  These image pairs had insufficient number of SIFT keypoints (<{:d}), will use ECC'.format(SIFT_nmatches_min))
            if method == 'SIFT-ECC' or method == 'ECC':
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'  Determining adjacent tile transformations using ECC')
                transformations_results0 = self.determine_transformations_ECC(target_pairs=target_pairs, **kwargs)
                if len(self.fls_anchors)>0:
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'  Determining anchor tile transformations using ECC')
                    transformations_results1 = self.determine_anchor_transformations_ECC(**kwargs)
                    
                    # prepare anchor info. we need to combide the input anchor positions with translations (transformations) that we determined for each tile relative to its anchor (if available).
                    valid_anchors_ECC = np.where(self.ECC_anchor_transformation_valid)[0]
                    if len(valid_anchors_ECC) > 0:
                        anchors_ECC = []
                        for valid_anchor_ECC in valid_anchors_ECC:
                            tx_ECC = -1.0 * (self.anchor_input_transformation_matrices[valid_anchor_ECC][0, 2] + self.ECC_anchor_transformation_matrices[valid_anchor_ECC][0, 2])
                            ty_ECC = -1.0 * (self.anchor_input_transformation_matrices[valid_anchor_ECC][1, 2] + self.ECC_anchor_transformation_matrices[valid_anchor_ECC][1, 2])
                            anchors_ECC.append([valid_anchor_ECC, tx_ECC, ty_ECC, default_anchor_weight])
                        kwargs['anchors'] = anchors_ECC

            tile_positions = self.solve_montage_stitching(**kwargs)
            if verbose:
                print('New Tile Positions:')
                print(tile_positions)
            tr_matr_new = self.update_translation_matrices(**kwargs)
            
            return tr_matr_new
    
    def assemble_montage(self, **kwargs):
        '''
        Assemble montage based on transformation matrices for each tile.
        
        kwargs:
        ----------
        weight_min : float
            vmin for weight. Default is 1
        weight_max : float
            vmax for weight. Default is 2048
        use_only_validated_transformations : boolean
            use only the tiles with validated transformation
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        verbose : boolean
            Display intermediate results. Default is False.
        
        '''
        use_only_validated_transformations = kwargs.get('use_only_validated_transformations', True)
        weight_min = kwargs.get('weight_min', 1.0)
        weight_max = kwargs.get('weight_max', 2048.0)
        DASK_client = kwargs.get('DASK_client', '')
        verbose = kwargs.get('verbose', False)
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=verbose)
        deformation_field = kwargs.get('deformation_field', np.nan)
        left_crop = kwargs.get('left_crop', 0)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        
        if (not any(self.valid_transformations>0)) and (not use_only_validated_transformations):
            if verbose:
                print('No valid transformation present. Fast-build montage using FirstPixels data')
            montage = self.assemble_montage_raw(**kwargs)
        else:
            montage = np.zeros((self.Ysize, self.Xsize-left_crop), dtype=float)
            montage_weights = np.zeros((self.Ysize, self.Xsize-left_crop), dtype=float)
            tile_params_mult = []
            for fl, (j, tr_matr_single) in zip(tqdm(self.fls, desc = 'Building tile parameter sets', display = verbose), enumerate(self.tr_matr)):
                add_tile  = (use_only_validated_transformations and self.valid_transformations[j]>0) or (not use_only_validated_transformations)
                if add_tile:
                    tile_params_mult.append([j, fl, tr_matr_single, self.Ysize, self.Xsize, weight_min, weight_max, left_crop])
            if len(tile_params_mult)>0:
                if use_DASK:
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Started DASK Computation')
                    shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                    futures = DASK_client.map(transform_tile, tile_params_mult, deformation_field=shared_data_future)
                    for future in as_completed(futures):
                        tile_out, weight_out, xi, xa, yi, ya = future.result()
                        montage[yi:ya, xi:xa] = montage[yi:ya, xi:xa] + tile_out
                        montage_weights[yi:ya, xi:xa] = montage_weights[yi:ya, xi:xa] + weight_out
                        future.cancel()
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Finished post-DASK Computation')
                else:
                    for tile_params in tqdm(tile_params_mult, desc = 'Building montage'):
                        if verbose:
                            print('Performing transform_tile with the following parameters:')
                            print(tile_params)
                        tile_out, weight_out, xi, xa, yi,  ya = transform_tile(tile_params, deformation_field)
                        if verbose:
                            print('Output is:')
                            print('tile_out.shape=', tile_out.shape, 'weight_out.shape=', weight_out.shape)
                            print('xi={:d}, xa={:d}, yi={:d},  ya={:d}'.format(xi, xa, yi,  ya))
                        montage[yi:ya, xi:xa] = montage[yi:ya, xi:xa] + tile_out
                        montage_weights[yi:ya, xi:xa] = montage_weights[yi:ya, xi:xa] + weight_out
            montage_weights = np.clip(montage_weights, weight_min, weight_max*np.product(self.shape)) 
            montage = montage / montage_weights
            self.montage = np.nan_to_num(montage, nan=-6000)
            self.montage_weights = montage_weights
        return self.montage
    
    def assemble_montage_raw(self, **kwargs):
        '''
        Assemble the montage based on FirstPixelX and FirstPixelY (shifts only) data for each tile.
        
        kwargs:
        ----------
        weight_min : float
            vmin for weight. Default is 1
        weight_max : float
            vmax for weight. Default is 2048
        verbose : boolean
            Display intermediate results. Default is False.
        '''
        weight_min = kwargs.get('weight_min', 1.0)
        weight_max = kwargs.get('weight_max', 2048.0)        
        verbose = kwargs.get('verbose', False)

        montage = np.zeros((self.Ysize, self.Xsize), dtype=float)
        montage_weights = np.zeros((self.Ysize, self.Xsize), dtype=float)

        for fl in tqdm(self.fls, desc = 'Building Raw Montage'):
            fr = FIBSEM_frame(fl)
            weight_loc = build_weight_array((fr.YResolution, fr.XResolution), weight_min = weight_min, weight_max = weight_max)
            xi = fr.FirstPixelX - self.FirstPixels[0, 0]
            xa = xi + fr.XResolution
            yi = fr.FirstPixelY - self.FirstPixels[0, 1]
            ya = yi + fr.YResolution
            montage[yi:ya, xi:xa] = montage[yi:ya, xi:xa] + fr.RawImageA * weight_loc
            montage_weights[yi:ya, xi:xa] = montage_weights[yi:ya, xi:xa] + weight_loc
        
        montage_weights = np.clip(montage_weights, weight_min, weight_max*np.product(self.shape)) 
        montage = montage / montage_weights
        self.montage = np.nan_to_num(montage, nan=-6000)
        self.montage_weights = montage_weights
        return self.montage

    def save_snapshot(self, **kwargs):
        '''
        Build an image that contains the montage and some data.

        kwargs:
        ----------
        verbose : boolean
            Display intermediate results. Default is False.
        overlay_tile_grid : boolean
            If True (Default), overlays tile grid.          
        thr_min : float
            Lower CDF threshold for determining the minimum data value. Default is 1.0e-3
        thr_max : float
            Upper CDF threshold for determining the maximum data value. Default is 1.0e-3
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion.
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion.
        nbins : int
            Number of histogram bins for building the PDF and CDF.
        verbose : boolean
            If True (default) display the results.
        dpi : int
            DPI. Default is 300.
        snapshot_name : string
            The name of the image to perform these operations (default is frame_name + '_snapshot.png').

        '''
        verbose = kwargs.get('verbose', False)
        overlay_tile_grid = kwargs.get('overlay_tile_grid', True)
        thr_min = kwargs.get('thr_min', 1.0e-3)
        thr_max = kwargs.get('thr_max', 1.0e-3)
        nbins = kwargs.get('nbins', 256)
        linestyle = kwargs.get('linestyle', 'dashed')
        linewidth = kwargs.get('linewidth', 0.25)
        fontsize = kwargs.get('fontsize', 10)
        color = kwargs.get('color', 'cyan')
        verbose = kwargs.get('verbose', True)
        dpi = kwargs.get('dpi', 300)
        data_dir = kwargs.get('data_dir', self.data_dir)
        snapshot_name = kwargs.get('snapshot_name', os.path.join(data_dir, self.fnm_montage.replace('.dat','_snapshot.png')))
        fig, axs = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios" : [1.5, 2]})
        fig.suptitle(snapshot_name, fontsize = fontsize)
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.15, hspace=0.1)
        if not hasattr(self, 'montage'):
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Raw montage have not been created, build it now')
            self.assemble_montage_raw(verbose=verbose)                        
        dmin, dmax = get_min_max_thresholds(self.montage, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
        axs[1].imshow(self.montage, cmap='Greys', vmin=dmin, vmax=dmax)
        axs[1].axis(False)
        if overlay_tile_grid:
            overlay_montage_grid(axs[1], self,
                             linewidth=linewidth,
                             linestyle=linestyle,
                             edgecolor=color)
        
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

        if self.FileVersion > 8:
            cell_text = [['Sample ID', '{:s}'.format(self.Sample_ID.strip('\x00')), '',
                          'Tile Size\n\nShape', '{:d} x {:d}\n\n{:d} x {:d}'.format(self.XResolution, self.YResolution, self.shape[1], self.shape[0]), '',
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
                              'Tile Size\n\nShape', '{:d} x {:d}\n\n{:d} x {:d}'.format(self.XResolution, self.YResolution, self.shape[1], self.shape[0]), '',
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
                cell_text = [['', '', '',
                              'Tile Size\n\nShape', '{:d} x {:d}\n\n{:d} x {:d}'.format(self.XResolution, self.YResolution, self.shape[1], self.shape[0]), '',
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
        
        fig.savefig(snapshot_name, dpi=dpi)
        
    def save_montage_dat(self, **kwargs):
        '''
        Save existing montage into a new .dat file. Only works on .dat files at the moment. ©G.Shtengel 11/2025 gleb.shtengel@gmail.com
         kwargs:
            ----------
            save_filename : string
                Filename for saving the padded frame. Default is os.path.join(self.data_dir, self.fnm_montage).
            fill_value : int
                Fill value for padding. Default is -6000.
            verbose : boolean
                Display intermediate comments / results. Default is False.
        '''
        save_filename = kwargs.get('save_filename', os.path.join(self.data_dir, self.fnm_montage))
        fill_value = kwargs.get('fill_value', -6000)
        verbose = kwargs.get('verbose', False)
        if not hasattr(self, 'montage'):
            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Raw montage have not been created, build it now')
            self.assemble_montage_raw(verbose=verbose)
        YResolution_new, XResolution_new = self.montage.shape
        #FirstPixelX_new, FirstPixelY_new = self.FirstPixels[0]
        if verbose:
            print('Output Frame Size: {:d} x {:d} pixels'.format(XResolution_new, YResolution_new))
            print('Data will be saved into the file: ', save_filename)
            print('Will use fill value = {:d}'.format(fill_value))

        # Update the header with new frame size information
        fr = FIBSEM_frame(self.fls[0], read_header_only=True)
        header = fr.header
        header_new = bytearray(header)
        XResolution_new_string =  pack('>L', XResolution_new)
        header_new[100:104] = XResolution_new_string
        YResolution_new_string =  pack('>L', YResolution_new)
        header_new[104:108] = YResolution_new_string
        ChanNum_new = 1
        ChanNum_new_string =  pack('b', ChanNum_new)
        header_new[32:33] = ChanNum_new_string
        AI2_new = 0
        AI2_new_string =  pack('b', AI2_new)
        header_new[152:153] = AI2_new_string
        '''
        FirstPixelX_new_string =  pack('>l', FirstPixelX_new)
        header_new[70:74] = FirstPixelX_new_string
        FirstPixelY_new_string =  pack('>l', FirstPixelY_new)
        header_new[74:78] = FirstPixelY_new_string
        '''
        # Create new Raw data array
        dt = np.dtype(np.int16).newbyteorder('>')

        # Save new frame
        with open(save_filename, 'wb') as f:
            f.write(header_new)
            self.montage.reshape(-1).astype(dt).tofile(f)
        return save_filename


class FIBSEM_montage_stack: 
    '''
    A class representing a stack of FIB-SEM montages - multiple z-panes consisting of multiple tiles.
    ©G.Shtengel 12/2025 gleb.shtengel@gmail.com
    Contains the info/settings on the FIB-SEM montage and the procedures that can be performed on it.
    '''
    
    def __init__(self, fls, **kwargs):
        '''
        Initializes (or recalls) an instance of  FIBSEM_montage_stack object. ©G.Shtengel 12/2025 gleb.shtengel@gmail.com

        Parameters:
        ----------
        fls : 2D array of str
            Filenames for the individual data frames in the stack of montages.

        kwargs:
        ---------
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif).
        
        memory_profiling : boolean
            Perform memory profiling during the data load and output it. Default is False.
        intralayer_weight : float, default 1.0
            Weight for pairwise constraints within a single Z-layer.
        interlayer_weight : float, default 100.0
            Weight for pairwise constraints for tiles between adjacent Z-layers.(100–10000 typical).
        add_reverse_edges : bool, default False
            If True, adds both (i->j) and (j->i) with same weight (increases robustness).
        anchor_ids : list or array of indices of the anchor tiles.
        EightBit : int
            If 1 then the data is assumed uint8, otherwise int16
        dump_filename : str
            Filename (full path) to a binary dump file with saved dataset attributes. If dump_filename points to a valid binary file the data set saved in that file will be recalled. Default is empty string ''.
        data_dir : str
            Data directory (path).
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3. Default is 3.
        Sample_ID : str
            Sample ID.
        PixelSize : float
            Pixel size in nm. Default is determined from the frame metadata. If that is not available, default is 8.0.
        Scaling : 2D array of floats
            Scaling parameters allowing to convert I16 data into actual electron counts.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is 1e-3.
        thr_max : float
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
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
            Matrix of regularization (shrinkage) parameters (applicable only if RegularizedAffineTransform is used). Default is 1e-5.
        targ_vector = 1D float array
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
            Min number of matches for the transformation to be considered valid. Delault is 5.
        int_order : int
            The order of interpolation (when transforming the data).
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        dtp : Data Type
            Python data type for saving. Default is np.int16.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        perform_deformation : boolean
            If True - the data is deformed (in addition to tyransformation defined above) using the deformation field data defined below.
        deformation_type : str
            Type of Deformation. Options are:
                'post_1DY'  - Default. Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
                'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
        deformation_sigma :  list of 1 or two floats.
            Gaussian width of smoothing (units of pixels). Default is 50.    
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
        self.data_dir = kwargs.get('data_dir', os.path.split(fls[0, 0])[0])
        self.ftype = kwargs.get('ftype', 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        self.intralayer_weight = kwargs.get('intralayer_weight', 1.0)
        self.interlayer_weight = kwargs.get('interlayer_weight', 100.0)
        self.add_reverse_edges = kwargs.get('add_reverse_edges', False)
        test_frame = FIBSEM_frame(fls[0, 0], ftype = self.ftype, calculate_scaled_images=False, read_header_only=True)

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
        self.Scaling = kwargs.get("Scaling", test_frame.Scaling)
        if hasattr(test_frame, 'PixelSize'):
            self.PixelSize = kwargs.get("PixelSize", test_frame.PixelSize)
        else:
            self.PixelSize = kwargs.get("PixelSize", 8.0)
        self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        self.DetA = test_frame.DetA
        self.DetB = test_frame.DetB
        self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        self.Sample_ID = kwargs.get("Sample_ID", test_frame.Sample_ID)
        self.EightBit = kwargs.get("EightBit", 1)
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
        self.max_iter = kwargs.get("max_iter", 1000)
        self.SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
        self.save_res_png  = kwargs.get("save_res_png", True)
        self.fnm_types = kwargs.get("fnm_types", ['mrc'])
        self.flipY = kwargs.get("flipY", False)                     # If True, the registered data will be flipped along Y axis
                                                                    # window size 701, polynomial order 3
        self.int_order = kwargs.get("int_order", False)             #     The order of interpolation. The order has to be in the range 0-5:
                                                                    #    - 0: Nearest-neighbor
                                                                    #    - 1: Bi-linear (default)
                                                                    #    - 2: Bi-quadratic
                                                                    #    - 3: Bi-cubic
                                                                    #    - 4: Bi-quartic
                                                                    #    - 5: Bi-quintic
        
        self.pad_edges =  kwargs.get("pad_edges", True)
        self.perform_deformation = kwargs.get("perform_deformation", False)
        self.deformation_type = kwargs.get("deformation_type", 'post_1DY')
        self.deformation_sigma = kwargs.get('deformation_sigma', 50)
        try:
            build_fnm_montage = os.path.splitext(os.path.split(fls[0, 0])[1])[0][0:-5] + 'montage_stack.mrc'
        except:
            build_fnm_montage = 'montage_stack.mrc'
        self.fnm_montage = kwargs.get("fnm_montage", build_fnm_montage)
        self.dtp = kwargs.get("dtp", np.int16)
        kwargs.update({'data_dir' : self.data_dir, 'fnm_montage' : self.fnm_montage, 'dtp' : self.dtp})
               
        FirstPixels = []
        for fl in fls[0]:
            fr = FIBSEM_frame(fl, read_header_only=True)
            FirstPixels.append([fr.FirstPixelX, fr.FirstPixelY])
        self.FirstPixels = np.array(FirstPixels)
        
        # try to auto-determine shape and adjacent pairs
        # self.nz_tiles  - # of layers (# of tiles along Z-axis)
        # self.ny_tiles  - # of rows per layer (# of tiles along Y-axis)
        # self.nx_tiles  - # of columns per layer(# of tiles along X-axis)
        self.nz_tiles, num_fls_zslice = self.fls.shape
        try:
            tile_string = os.path.splitext(os.path.split(self.fls[0, -1])[1])[0][-5:].split('-')
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
        self.Xoverlap = self.XResolution - (self.FirstPixels[1, 0] - self.FirstPixels[0, 0])
        self.Yoverlap = self.YResolution - (self.FirstPixels[self.shape[1], 1] - self.FirstPixels[(self.shape[1]-1), 1])

        # create the structure for pairwice tile transformation
        L = self.nz_tiles
        M = self.ny_tiles
        N = self.nx_tiles
        V = L * M * N                     # Total number of tiles
        nh = L * M * (N - 1)              # Total number of left-right intra-layer pairs
        nv = L * (M - 1) * N              # Total number of up-down intra-layer pairs
        nl = (L - 1) * M * N              # Total number of inter-layer pairs
        C = nh + nv + nl                  # Total number of of pairs (pair-wise translations)
        # horiz_trans: np.ndarray (L, M, N-1, 2), translations to right neighbor (x,y)
        # vert_trans: np.ndarray (L, M-1, N, 2), translations to bottom neighbor (x,y)
        # layer_trans: np.ndarray (L-1, M, N, 2), translations to upper layer (x,y)

        w_sqrt_intra = np.sqrt(self.intralayer_weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||
        w_sqrt_inter = np.sqrt(self.interlayer_weight)

        # Prepare data for sparse matrix A
        data = []
        row_ind = []
        col_ind = []
        row = 0   # row (entry) in the sparse matrix A (not a tile row)

        # Build a sparse matrix A for Ax=b lsqr equation
        # idx1 and idx2 are absolute (in 1D sense) tile indecis
        # each entry is a single sparse matrix element, there are two elements per pairwise translation condtion, they enter with opposite signs

        # Horizontal adjacent pairs (intra-layer)
        for l in range(L):
            for i in range(M):
                for j in range(N - 1):
                    idx1 = l * M * N + i * N + j
                    idx2 = l * M * N + i * N + j + 1
                    row_ind.extend([row, row])
                    col_ind.extend([idx1, idx2])
                    data.extend([-w_sqrt_intra, w_sqrt_intra])
                    row += 1

        # Vertical adjacent pairs (intra-layer)
        for l in range(L):
            for i in range(M - 1):
                for j in range(N):
                    idx1 = l * M * N + i * N + j
                    idx2 = l * M * N + (i + 1) * N + j
                    row_ind.extend([row, row])
                    col_ind.extend([idx1, idx2])
                    data.extend([-w_sqrt_intra, w_sqrt_intra])
                    row += 1

        # Layer-to-layer correspondences (inter-layer)
        for l in range(L - 1):
            for i in range(M):
                for j in range(N):
                    idx1 = l * M * N + i * N + j
                    idx2 = (l + 1) * M * N + i * N + j
                    row_ind.extend([row, row])
                    col_ind.extend([idx1, idx2])
                    data.extend([-w_sqrt_inter, w_sqrt_inter])
                    row += 1

        self.A_csr = csr_matrix((data, (row_ind, col_ind)), shape=(C, V)) # sparse matrix

        self.pair_indices = np.array(col_ind).reshape((row, 2))   # absolute (in 1D sense) tile indecis for each pair
        self.pair_margins = [[self.YResolution, 2*self.Xoverlap] for x in np.arange(nh)] + [[2*self.Yoverlap, self.XResolution] for x in np.arange(nv)] + [[self.YResolution, self.XResolution] for x in np.arange(nl)]
        eye3x3 = np.eye(3,3)
        self.ECC_transformation_matrices = np.repeat(eye3x3[np.newaxis, :, :], C, axis=0)
        self.ECC_transformation_valid = np.full(C, False)
        self.SIFT_transformation_matrices = np.repeat(eye3x3[np.newaxis, :, :], C, axis=0)
        self.SIFT_transformation_valid = np.full(C, False)
        self.SIFT_fnms_matches = ['' for x in np.arange(C)]
        self.SIFT_nmatches = np.full(C, 0)

        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Initialized FIBSEM_montage_stack instance:')
            print('Total number of tile files: {:d}'.format(V))
            print('Individual Z-slice shape (nx_tiles, ny_tiles): {:d} x {:d} tiles'.format(self.nx_tiles, self.ny_tiles))
            print('Number of Z-slices (nz_tiles): {:d}'.format(self.nz_tiles))
            print('Total number of pairwise transformations : {:d}'.format(C))
    
        # initialize the montage size (assuming rectangular shape)
        self.Xsize = self.shape[1] * (self.XResolution - self.Xoverlap) + self.Xoverlap
        self.Ysize = self.shape[0] * (self.YResolution - self.Yoverlap) + self.Yoverlap
        
        # initialize the translation matrix for each tile
        shifts_x = self.FirstPixels[:, 0] - self.FirstPixels[0, 0]
        shifts_y = self.FirstPixels[:, 1] - self.FirstPixels[0, 1]
        single_layer_tr_matr = np.repeat(eye3x3[np.newaxis, :, :], M * N, axis=0)
        single_layer_tr_matr[:, 0, 2] = - np.array(shifts_x).flatten()
        single_layer_tr_matr[:, 1, 2] = - np.array(shifts_y).flatten()
        self.tr_matr = np.repeat(single_layer_tr_matr[np.newaxis, :, :, :], L, axis=0)

        if memory_profiling:
            elapsed_time = elapsed_since(start_time)
            rss_after, vms_after, shared_after = get_process_memory()
            print("Profiling: Start of Execution: RSS: {:>8} | VMS: {:>8} | SHR {"
                  ":>8} | time: {:>8}"
                .format(format_bytes(rss_after - rss_before),
                        format_bytes(vms_after - vms_before),
                        format_bytes(shared_after - shared_before),
                        elapsed_time))


    def evaluate_FIBSEM_statistics(self, **kwargs):
        '''
        Evaluates parameters of FIBSEM montage (Min/Max, Working Distance (WD), Milling Y Voltage (MV), FOV center positions). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        frame_inds : array
            Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames.
        data_dir : str
            Data directory (path). Default is object attribute.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        FIBSEM_Data_xlsx : str
            File path of the Excell file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).
        use_existing_data : boolean
            Default is False. If True and the data exists (saved into XLSX), use that.            
        verbose : boolean
            If True, intermediate messages and results will be displayed. Default is False.

        Returns:
        list of 14 parameters: FIBSEM_Data_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions
            FIBSEM_Data_xlsx : str
                path to Excel file with the FIBSEM data
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8)
            data_man_glob : float   
                max data value for I8 conversion (open CV SIFT requires I8)
            center_x : float array
                FOV Center X-coordinate extracted from the header data
            center_y : float array
                FOV Center Y-coordinate extracted from the header data
            ScanRate : float array
                SEM Scan Rate (Hz)
            EHT : float array
                SEM EHT voltage (kV)
            SEMSpecimenI : float array
                SEM Specimen current (nA)
            XResolutions : int array
                X-frame sizes
            YResolutions : int array
                Y-frame sizes
        '''
        verbose = kwargs.get('verbose', False)
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        frame_inds = kwargs.get("frame_inds", np.arange(len(self.fls.ravel())))
        data_dir = kwargs.get('data_dir', self.data_dir)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        fit_params = kwargs.get('fit_params', ['SG', 3, 1])
        FIBSEM_Data_xlsx_default = os.path.join(data_dir, os.path.splitext(self.fnm_montage)[0] + '_FIBSEM_Data.xlsx')
        FIBSEM_Data_xlsx = kwargs.get('FIBSEM_Data_xlsx', FIBSEM_Data_xlsx_default)
        use_existing_data = kwargs.get('use_existing_data', False)

        local_kwargs = {'use_DASK' : use_DASK,
                        'DASK_client_retries' : DASK_client_retries,
                        'ftype' : ftype,
                        'frame_inds' : frame_inds,
                        'data_dir' : data_dir,
                        'thr_min' : thr_min,
                        'thr_max' : thr_max,
                        'nbins' : nbins,
                        'sliding_minmax' : False,
                        'fit_params' : fit_params,
                        'FIBSEM_Data_xlsx' : FIBSEM_Data_xlsx,
                        'verbose' : verbose,
                        'use_existing_data' : use_existing_data}

        if verbose:
            print('Evaluating the parameters of FIBSEM data set (data Min/Max, Working Distance, FOV center positions, Scan Rate, EHT)')
        self.FIBSEM_Data = evaluate_FIBSEM_frames_dataset(self.fls.ravel(), DASK_client, **local_kwargs)
        self.data_minmax = self.FIBSEM_Data[0:5]
        self.data_min_glob = self.FIBSEM_Data[1]
        self.data_max_glob = self.FIBSEM_Data[2]
        
        self.FOV_x = self.FIBSEM_Data[7]
        self.FOV_y = self.FIBSEM_Data[8]
        try:
            self.XResolutions = self.FIBSEM_Data[12].astype(int)
            self.YResolutions = self.FIBSEM_Data[13].astype(int)
        except:
            self.XResolutions = np.full(len(WD), self.XResolution).astype(int)
            self.YResolutions = np.full(len(WD), self.YResolution).astype(int)

        self.XResolution = np.max(self.XResolutions)
        self.YResolution = np.max(self.YResolutions)

        return self.FIBSEM_Data
    

    def extract_keypoints(self, **kwargs):
        '''
        Extract Key-Points and Descriptors. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. DASK client. If empty string '' (Default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
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
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is np.nan - no distortion correction
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        data_minmax : list of 5 parameters
            minmax_xlsx : str
                path to Excel file with Min/Max data.
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8).
            data_min_sliding : float array
                min data values (one per file) for I8 conversion.
            data_max_sliding : float array
                max data values (one per file) for I8 conversion.
            data_minmax_glob : 2D float array
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
        deformation_field : 2D array
             Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
        deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        interpolation : int
            Interpolation type as defined in CV2 (if deformation_field is not np.nan) . Default is cv2.INTER_LINEAR.
        fill_value = 0.0
            Fill value for outside pixeld in cv2.remap. Default is 0.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
        If True, outputs will be printed.
    
        Returns:
        fnms_kpts : array of str
            Filenames for binary files containing Key-Points and Descriptors for each frame.
        '''
        verbose = kwargs.get('verbose', True)
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = verbose)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        data_minmax = kwargs.get("data_minmax", self.data_minmax)
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        deformation_field = kwargs.get('deformation_field', np.nan)
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        fill_value = kwargs.get('fill_value', 0)
        use_existing_data = kwargs.get('use_existing_data', False)

        minmax_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
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
                    'fill_value' : fill_value}

        params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls.ravel()]        
        if use_DASK:
            shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
            futures_s3 = DASK_client.map(extract_keypoints_descr_files, params_s3, deformation_field = shared_data_future, retries = DASK_client_retries)
            fnms_kpts = DASK_client.gather(futures_s3)
        else:
            fnms_kpts = []
            for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ', display=verbose)):
                fnms_kpts.append(extract_keypoints_descr_files(param_s3, deformation_field))
        self.fnms_kpts = np.array(fnms_kpts).reshape(self.fls.shape)
        return fnms_kpts
    

    def determine_transformations_SIFT(self, **kwargs):
        '''
        Determine transformation matrices for frame pairs using SIFT. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        pair_margins : arraiy of tuples of 2 ints
            Parts of images to be used. It is assumed that first image (img1) in each target_pair is to the left and above of the second image (img2).
            Subsets img1[-ymargin:, :] and  img2[0:ymargin, :] or img1[:, -xmargin:] and  img2[:, 0:xmargin] will be used for correlation.
            Default is full images, so image_margins = (self.YResolution, self.XResolution)
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        TransformType : object reference
            Transformation model used for determining the transformation matrix from Key-Point pairs. Default is object attribute.
            Choose from the following options:
                ShiftTransform - only x-shift and y-shift
                XScaleShiftTransform  -  x-scale, x-shift, y-shift
                ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
                AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
                RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
        l2_matrix : 2D float array
           Matrix of regularization (shrinkage) parameters (applicable only if RegularizedAffineTransform is used). Default is object attribute.
        targ_vector = 1D float array
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
            Min number of matches for the transformation to be considered valid. Delault is 5.
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
        transformations_results : array of lists containing the results:
            [transformation_matrix, fnm_matches, npt, error_abs_mean, error_FWHMx, error_FWHMy, iteration]
            transformation_matrix : 2D float array
                Transformation matrix for each sequential frame pair.
            fnm_matches : str
                Filename containing the matches used to determine the transformation for the pair of frames.
            npts : int
                Number of matches.
            error_abs_mean : float
                Mean abs error of registration for all matched Key-Points.
        '''
        verbose = kwargs.get('verbose', False)
        if len(self.fnms_kpts) == 0:
            if verbose:
                print('No data on individual key-point data files, peform key-point search')
            transformations_results = []
        else:
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            ftype = kwargs.get("ftype", self.ftype)
            deformation_field = kwargs.get('deformation_field', np.nan)
            left_crop = kwargs.get('left_crop', 0)
            if hasattr(self, 'pair_margins'):
                pair_margins = kwargs.get('pair_margins', self.pair_margins)
            else:
                pair_margins = kwargs.get('pair_margins', (self.YResolution, self.XResolution))
            TransformType = kwargs.get("TransformType", self.TransformType)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            if hasattr('self', 'SIFT_nmatches_min'):
                SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
            else:
                SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
            Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            save_res_png  = kwargs.get("save_res_png", self.save_res_png )
            start = kwargs.get('start', 'edges')
            estimation = kwargs.get('estimation', 'interval')
            use_existing_data = kwargs.get('use_existing_data', False)

            params_SIFT = []
            fnms_kpts = self.fnms_kpts.ravel()

            for index_pair, pair_margins  in zip(tqdm(self.pair_indices, desc='Setting up SIFT parameter list', display=verbose), self.pair_margins):
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
                        'verbose' : verbose}

                fname1 = fnms_kpts[index_pair[0]]
                fname2 = fnms_kpts[index_pair[1]]
                path_base, f1 = os.path.split(fname1)
                _, f2 = os.path.split(fname2)
                fnm_matches = os.path.join(path_base, f1.replace('_kpdes.bin', '_')+f2.replace('_kpdes.bin', '_matches.bin'))
                dt_kwargs['fnm_matches'] = fnm_matches
                index_loc0, index_loc1 = np.mod(index_pair, self.nx_tiles*self.ny_tiles)
                FirstPixels_delta = self.FirstPixels[index_loc1] - self.FirstPixels[index_loc0]
                ymargin, xmargin = pair_margins
                dt_kwargs['warp_matrix'] = np.array([[1, 0, -FirstPixels_delta[0]], [0, 1, -FirstPixels_delta[1]]], dtype=np.float32)
                dt_kwargs['image_margins'] = (ymargin, xmargin)
                dt_kwargs['image_shape'] = (self.YResolution, self.XResolution)
                dt_kwargs['left_crop'] = left_crop
                param_SIFT = [fname1, fname2, dt_kwargs]
                params_SIFT.append(param_SIFT)
                if verbose:
                    print('Added a set: ')
                    print([fname1, fname2, dt_kwargs])

            if use_DASK:
                futures_SIFT = DASK_client.map(determine_transformations_files, params_SIFT, retries = DASK_client_retries)                
                transformations_results_3D = DASK_client.gather(futures_SIFT)
            else:
                transformations_results_3D = []
                for param_SIFT in tqdm(params_SIFT, desc = 'Extracting Transformation Parameters: ', display=verbose):
                    transformations_results_3D.append(determine_transformations_files(param_SIFT))
            
            for j, transformations_result  in enumerate(tqdm(transformations_results_3D, desc = 'Parsing the SIFT results', display = verbose)):
                try:
                    self.SIFT_transformation_matrices[j] = np.nan_to_num(transformations_result[0])
                    self.SIFT_fnms_matches[j] = transformations_result[1]
                    self.SIFT_nmatches[j] = len(transformations_result[2][0])
                    self.SIFT_transformation_valid[j] = self.SIFT_nmatches[j] > SIFT_nmatches_min

                except Exception as e:
                    if verbose:
                        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   An error occurred: {}'.format(e))
                        print('transformations_result:  ', transformations_result)
            if verbose:
                L = self.nz_tiles
                M = self.ny_tiles
                N = self.nx_tiles
                V = L * M * N                     # Total number of tiles
                nh = L * M * (N - 1)              # Total number of left-right intra-layer pairs
                nv = L * (M - 1) * N              # Total number of up-down intra-layer pairs
                nl = (L - 1) * M * N              # Total number of inter-layer pairs
                C = nh + nv + nl                  # Total number of of pairs (pair-wise translations)
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for intra-layer horisontal matches :', np.mean(self.SIFT_nmatches[0:nh]).astype(np.int64))
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for intra-layer vertical matches :', np.mean(self.SIFT_nmatches[nh:nh+nv]).astype(np.int64))
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for inter-layer matches :', np.mean(self.SIFT_nmatches[nh+nv:]).astype(np.int64))

                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Valid SIFT transformation established: ', self.SIFT_transformation_valid)
        return transformations_results_3D

    def test_SIFT(self, index_pair, pair_margins, **kwargs):
        ftype = kwargs.get("ftype", self.ftype)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        data_minmax = kwargs.get("data_minmax", self.data_minmax)
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        deformation_field = kwargs.get('deformation_field', np.nan)
        interpolation = kwargs.get('interpolation', cv2.INTER_LINEAR)
        fill_value = kwargs.get('fill_value', 0)

        TransformType = kwargs.get("TransformType", self.TransformType)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        if hasattr('self', 'SIFT_nmatches_min'):
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
        else:
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
        Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        start = kwargs.get('start', 'edges')
        estimation = kwargs.get('estimation', 'interval')
        verbose = kwargs.get('verbose' : True)

        minmax_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
        kpt_kwargs = {'ftype' : ftype,
                    'thr_min' : thr_min,
                    'thr_max' : thr_max,
                    'nbins' : nbins,
                    'SIFT_nfeatures' : SIFT_nfeatures,
                    'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
                    'SIFT_contrastThreshold' : SIFT_contrastThreshold,
                    'SIFT_edgeThreshold' : SIFT_edgeThreshold,
                    'SIFT_sigma' : SIFT_sigma,
                    'use_existing_data' : False,
                    'save_deformed_image' : True,
                    'interpolation' : interpolation,
                    'fill_value' : fill_value,
                    'verbose' : verbose}

        fl1 = self.fls.ravel()[index_pair[0]]
        fl2 = self.fls.ravel()[index_pair[1]]
        params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in [fl1, fl2]]
        fnms_kpts = []
        for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ', display=verbose)):
            fnms_kpts.append(extract_keypoints_descr_files(param_s3, deformation_field))

        params_SIFT = []
            fnms_kpts = self.fnms_kpts.ravel()

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

        fnm_deformed1 = os.path.splitext(fname1)[0] + '_def_image.tif'
        fnm_deformed2 = os.path.splitext(fname2)[0] + '_def_image.tif'
        path_base, f1 = os.path.split(fname1)
        _, f2 = os.path.split(fname2)
        fnm_matches = os.path.join(path_base, f1.replace('_kpdes.bin', '_')+f2.replace('_kpdes.bin', '_matches.bin'))
        dt_kwargs['fnm_matches'] = fnm_matches
        index_loc0, index_loc1 = np.mod(index_pair, self.nx_tiles*self.ny_tiles)
        FirstPixels_delta = self.FirstPixels[index_loc1] - self.FirstPixels[index_loc0]
        ymargin, xmargin = pair_margins
        dt_kwargs['warp_matrix'] = np.array([[1, 0, -FirstPixels_delta[0]], [0, 1, -FirstPixels_delta[1]]], dtype=np.float32)
        dt_kwargs['image_margins'] = (ymargin, xmargin)
        dt_kwargs['image_shape'] = (self.YResolution, self.XResolution)
        dt_kwargs['left_crop'] = left_crop
        param_SIFT = [fname1, fname2, dt_kwargs]

        transformations_result = determine_transformations_files(param_SIFT)

        print('SIFT_transformation_matrix = ', transformations_result[0])
        print('SIFT_fnms_matches: ', transformations_result[1])
        print('SIFT_nmatches = ', len(transformations_result[2][0]))

        return fnm_deformed1, fnm_deformed2, transformations_result

    def determine_transformations_ECC(self, **kwargs):
        '''
        Determine transformation matrices for frame pairs using ECC. ©G.Shtengel 12/2025 gleb.shtengel@gmail.com
        Uses find_Transform_ECC(img1, img2, **kwargs).
        
        kwargs:
        ---------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        pair_margins : arraiy of tuples of 2 ints
            Parts of images to be used. It is assumed that first image (img1) in each target_pair is to the left and above of the second image (img2).
            Subsets img1[-ymargin:, :] and  img2[0:ymargin, :] or img1[:, -xmargin:] and  img2[:, 0:xmargin] will be used for correlation.
            Default is full images, so image_margins = (self.YResolution, self.XResolution)
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        motion : target transformation.
            Default is cv2.MOTION_TRANSLATION
        repeats : int
            repeat internally this many times. Default is 2.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
        verbose : boolean
            Display intermediate results. Default is True.
        
        Returns:
        transformations_results : array of lists containing the results:
            [transformation_matrix, error_code]
            transformation_matrix : 2D float array
                Transformation matrix for each sequential frame pair.
            error_code : int
                CV2 error code.
        '''
        ftype = kwargs.get('ftype', self.ftype)
        repeats = kwargs.get('repeats', 2)
        verbose = kwargs.get('verbose', False)
        use_existing_data = kwargs.get('use_existing_data', False)
        deformation_field = kwargs.get('deformation_field', np.nan)
        left_crop = kwargs.get('left_crop', 0)
        if hasattr(self, 'pair_margins'):
            pair_margins = kwargs.get('pair_margins', self.pair_margins)
        else:
            pair_margins = kwargs.get('pair_margins', (self.YResolution, self.XResolution))
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
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        use_existing_data = kwargs.get('use_existing_data', False)
        params_ECC = []
        fls = self.fls.ravel()

        for index_pair, pair_margins  in zip(tqdm(self.pair_indices, desc='Setting up ECC parameter list', display=verbose), self.pair_margins):
            dt_kwargs = {'ftype' : ftype,
                     'motion' : motion,
                     'criteria' : criteria,
                     'use_existing_data' : use_existing_data,
                     'verbose' : verbose}
            fname1 = fls[index_pair[0]]
            fname2 = fls[index_pair[1]]
            index_loc0, index_loc1 = np.mod(index_pair, self.nx_tiles*self.ny_tiles)
            FirstPixels_delta = self.FirstPixels[index_loc1] - self.FirstPixels[index_loc0]
            ymargin, xmargin = pair_margins
            dt_kwargs['warp_matrix'] = np.array([[1, 0, -FirstPixels_delta[0]], [0, 1, -FirstPixels_delta[1]]], dtype=np.float32)
            dt_kwargs['image_margins'] = (ymargin, xmargin)
            dt_kwargs['left_crop'] = left_crop
            param_ECC = [fname1, fname2, dt_kwargs]
            params_ECC.append(param_ECC)
        if use_DASK:
            shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
            futures_ECC = DASK_client.map(find_Transform_ECC_DASK, params_ECC, deformation_field = shared_data_future, retries = DASK_client_retries)
            transformations_results_3D = DASK_client.gather(futures_ECC)
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
        Solve montage stack stitching (perform bundle optimization). ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        verbose : boolean
            Display intermediate results. Default is True.
        method : string
            Options are: ['SIFT-ECC', 'SIFT', 'ECC']. Default is 'ECC'.  'SIFT-ECC' means - try SIFT first, and for the tiles that SIFT failed, try ECC.
        
        Returns:
        positions : array of new tile positions.
        '''
        
        verbose = kwargs.get('verbose', False)
        method = kwargs.get('method', 'ECC')
        valid_methods = ['SIFT-ECC', 'SIFT', 'ECC']

        L = self.nz_tiles
        M = self.ny_tiles
        N = self.nx_tiles
        V = L * M * N                     # Total number of tiles
        nh = L * M * (N - 1)              # Total number of left-right intra-layer pairs
        nv = L * (M - 1) * N              # Total number of up-down intra-layer pairs
        nl = (L - 1) * M * N              # Total number of inter-layer pairs
        C = nh + nv + nl                  # Total number of of pairs (pair-wise translations)
        # horiz_trans: np.ndarray (L, M, N-1, 2), translations to right neighbor (x,y)
        # vert_trans: np.ndarray (L, M-1, N, 2), translations to bottom neighbor (x,y)
        # layer_trans: np.ndarray (L-1, M, N, 2), translations to upper layer (x,y)

        # We already have self.A_csr = csr_matrix((data, (row_ind, col_ind)), shape=(C, V)) # sparse matrix
        # Now we need to contruct the matrix B and solve LSQ

        w_sqrt_intra = np.sqrt(self.intralayer_weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||
        w_sqrt_inter = np.sqrt(self.interlayer_weight)
        weights = np.concatenate((np.full((nh+nv), w_sqrt_intra), np.full(nl, w_sqrt_inter)))

        if method not in valid_methods:
            if verbose:
                print('Method ' + method +' is not among valid methods: ', valid_methods)
            return np.nan
        else:
            if method == 'SIFT':
                bx = self.SIFT_transformation_matrices[:, 0, 2] * weights
                by = self.SIFT_transformation_matrices[:, 1, 2] * weights
                res_x_all = lsqr(self.A_csr[self.SIFT_transformation_valid], bx[self.SIFT_transformation_valid])
                res_y_all = lsqr(self.A_csr[self.SIFT_transformation_valid], by[self.SIFT_transformation_valid])
            else:
                bx = self.ECC_transformation_matrices[:, 0, 2] * weights
                by = self.ECC_transformation_matrices[:, 1, 2] * weights
                res_x_all = lsqr(self.A_csr[self.ECC_transformation_valid], bx[self.ECC_transformation_valid])
                res_y_all = lsqr(self.A_csr[self.ECC_transformation_valid], by[self.ECC_transformation_valid])
        res_x = res_x_all[0]
        res_y = res_y_all[0]
        positions = np.zeros((V, 2))
        positions[:, 0] = res_x - res_x[0]
        positions[:, 1] = res_y - res_y[0]

        self.tr_matr[:, :, 0:2, 2] = positions.reshape((L, M*N, 2))
        self.tile_positions = -positions.reshape((L, M*N, 2))

        return self.tile_positions


    def assemble_layer_mosaic(self, layer_id, **kwargs):
        '''
        Assemble layer montage based on transformation matrices for each tile. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com

        Parameters:
        layer_id : int
            Layer ID should be a value bewteen -1 and self.nz_tiles-1. -1 means the last layer will be assembled.
        
        kwargs:
        ----------
        weight_min : float
            vmin for weight. Default is 1
        weight_max : float
            vmax for weight. Default is 2048
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        fill_value : int
            The value to assign to pixels outside the transformed image bounds. Default is -10000.
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        verbose : boolean
            Display intermediate results. Default is False.
        
        '''
        if layer_id<-1 or layer_id>self.nz_tiles-1:
            print('layer_id parameter {:d} is out of range: -1 to {:d}'.format(layer_id, self.nz_tiles))
            return np.nan

        DASK_client = kwargs.get('DASK_client', '')
        weight_min = kwargs.get('weight_min', 1.0)
        weight_max = kwargs.get('weight_max', 2048.0) 
        fill_value = kwargs.get ('fill_value', -10000) 
        verbose = kwargs.get('verbose', False)
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=verbose)
        deformation_field = kwargs.get('deformation_field', np.nan)
        left_crop = kwargs.get('left_crop', 0)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3) 

        layer_mosaic = np.zeros((self.Ysize, self.Xsize-left_crop), dtype=float)
        layer_mosaic_weights = np.zeros((self.Ysize, self.Xsize-left_crop), dtype=float)
        tile_params_mult = []
        xy_limits = []
        for fl, (j, tr_matr_single) in zip(tqdm(self.fls[layer_id], desc = 'Building tile parameter sets', display = verbose), enumerate(self.tr_matr[layer_id])):
            tile_params_mult.append([j, fl, tr_matr_single, self.Ysize, self.Xsize, weight_min, weight_max, left_crop])
        if len(tile_params_mult)>0:
            if use_DASK:
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Started DASK Computation')
                shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                futures = DASK_client.map(transform_tile, tile_params_mult, deformation_field=shared_data_future)
                for future in as_completed(futures):
                    tile_out, weight_out, xi, xa, yi, ya = future.result()
                    xy_limits.append([xi, xa, yi, ya])
                    layer_mosaic[yi:ya, xi:xa] = layer_mosaic[yi:ya, xi:xa] + tile_out
                    layer_mosaic_weights[yi:ya, xi:xa] = layer_mosaic_weights[yi:ya, xi:xa] + weight_out
                    future.cancel()
                if verbose:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Finished post-DASK Computation')
            else:
                for tile_params in tqdm(tile_params_mult, desc = 'Building mosaic for layer_id={:d}'.format(layer_id), display = verbose):
                    if verbose:
                        print('Performing transform_tile with the following parameters:')
                        print(tile_params)
                    tile_out, weight_out, xi, xa, yi,  ya = transform_tile(tile_params, deformation_field)
                    xy_limits.append([xi, xa, yi, ya])
                    if verbose:
                        print('Output is:')
                        print('tile_out.shape=', tile_out.shape, 'weight_out.shape=', weight_out.shape)
                        print('xi={:d}, xa={:d}, yi={:d},  ya={:d}'.format(xi, xa, yi,  ya))
                    layer_mosaic[yi:ya, xi:xa] = layer_mosaic[yi:ya, xi:xa] + tile_out
                    layer_mosaic_weights[yi:ya, xi:xa] = layer_mosaic_weights[yi:ya, xi:xa] + weight_out
            layer_mosaic_weights = np.clip(layer_mosaic_weights, weight_min, weight_max*np.product(self.shape)) 
            layer_mosaic = np.nan_to_num(layer_mosaic / layer_mosaic_weights, nan=-fill_value)
        return layer_mosaic, layer_mosaic_weights, xy_limits


    def save_stack(self, **kwargs):
        '''
        Assemble all layers based on transformation matrices for each tile and save them into stack. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        fnm_montage : string
            Silename to save the data. Default is object attribute self.fnm_montage
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is ['mrc']. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        voxel_size : rec array of 3 elemets
            voxel size in nm
        dtp  : dtype
            Python data type for saving. Default is int16, the other option currently is uint8.
        weight_min : float
            vmin for weight. Default is 1
        weight_max : float
            vmax for weight. Default is 2048
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        fill_value : int
            The value to assign to pixels outside the transformed image bounds. Default is -10000.
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        verbose : boolean
            Display intermediate results. Default is False.
        
        '''

        DASK_client = kwargs.get('DASK_client', '')
        fnm_montage = kwargs.get('fnm_montage', self.fnm_montage)
        fnm_types = kwargs.get("fnm_types", ['mrc'])
        voxel_size_default = np.rec.array((8.0, 8.0, 8.0), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        voxel_size = kwargs.get("voxel_size", voxel_size_default)
        voxel_size_angstr = voxel_size.copy()
        voxel_size_angstr.x = voxel_size_angstr.x * 10.0
        voxel_size_angstr.y = voxel_size_angstr.y * 10.0
        voxel_size_angstr.z = voxel_size_angstr.z * 10.0
        dtp = kwargs.get("dtp", np.int16)
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
        weight_min = kwargs.get('weight_min', 1.0)
        kwargs['weight_min'] = weight_min 
        weight_max = kwargs.get('weight_max', 2048.0)
        kwargs['weight_max'] = weight_max 
        fill_value = kwargs.get ('fill_value', -10000)
        kwargs['fill_value'] = fill_value
        verbose = kwargs.get('verbose', False)
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=verbose)
        deformation_field = kwargs.get('deformation_field', np.nan)
        DF0 = convert_tr_matr_into_deformation_field(np.eye(3,3).astype(float), (self.YResolution, self.XResolution))
        kwargs['deformation_field'] = deformation_field - DF0

        left_crop = kwargs.get('left_crop', 0)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)

        fnms_saved = []
        if 'mrc' in fnm_types:
            mrc_filename = os.path.splitext(fnm_montage)[0] + '.mrc'
            fnms_saved.append(mrc_filename)
            mrc_new = mrcfile.new_mmap(mrc_filename, shape=(self.nz_tiles, self.Ysize, self.Xsize), mrc_mode=mrc_mode, overwrite=True)
            mrc_new.voxel_size = voxel_size_angstr
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the registered stack into the file: ', mrc_filename)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
            for layer_id in tqdm(np.arange(self.nz_tiles), desc = 'Saving the data stack into MRC file'):
                mrc_new.data[layer_id, :, :] = self.assemble_layer_mosaic(layer_id, **kwargs)[0].astype(dtp)

        mrc_new.close()

        return fnms_saved

        