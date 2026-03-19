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
try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff

from dask.distributed import Client
from dask.distributed import as_completed
from IPython.display import IFrame
from ClusterWrap.clusters import janelia_lsf_cluster

from scipy.signal import savgol_filter
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.ndimage import map_coordinates

from FIBSEM_gs_py.FIBSEM_gs import (FIBSEM_frame,
                        ShiftTransform,
                        XScaleShiftTransform,
                        ScaleShiftTransform,
                        RegularizedAffineTransform,
                        get_min_max_thresholds,
                        extract_keypoints_descr_files,
                        extract_image_intensity, 
                        determine_transformations_files,
                        convert_tr_matr_into_deformation_field,
                        evaluate_FIBSEM_frames_dataset)

from FIBSEM_gs_py.FIBSEM_help_functions_gs import (check_DASK,
                                                    find_FWHM,
                                                    dask_remove_file,
                                                    elapsed_since,
                                                    get_process_memory,
                                                    format_bytes,
                                                    read_kwargs_xlsx,
                                                    parse_metadata_file)


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

    Returns:
    ----------
    weights
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
    tile_params : list :  j, fl, image_name, tr_matr_single, montage_ysz, montage_xsz, weight_min, weight_max
        j : int, tile ID
        fl : str, filename for the tile
        image_name : str, image name ('RawImageA' or 'RawImageB')
        tr_matr_single : 3x3 array : transformation matrix
        montage_xsz : int : montage x-size in pixels
        montage_ysz : int : montage y-size in pixels
        weight_min : float :  weight_min for weight
        weight_max : float :  weight_max for weight

    deformation_field : 3D array
        Deformation field for distortion corrections to be executed before ECC. Default is np.nan - no distortion correction

    Returns:
    ----------
    tile_out, weight_out, xi, xa-left_crop-x0, yi,  ya-y0
    
    '''
    j, fl, image_name, tr_matr_single, montage_ysz, montage_xsz, weight_min, weight_max, left_crop = tile_params
    fr = FIBSEM_frame(fl)
    if image_name == 'RawImageB':
        tile_initial = fr.RawImageB
    else:
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
        X-size of the tile. Default is montage_object.XResolution-left_crop
    dy : int
        Y-size of the tile. Default is montage_object.YResolution
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
    
    
    Returns:
    ---------- 
    image_deformed, shift_x, shift_y
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
    deformation_field : 2D array
        Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.
    
    ----------
    kwargs:
    interpolation : int
        Interpolation type as defined in CV2. Default is cv2.INTER_LINEAR.
    fill_value = 0.0
        Fill value for outside pixels in cv2.remap. Default is 0.
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


def assemble_layer(params, deformation_field):
    '''
    Assembles layer. Worker function called by assemble_layer_mosaic and save_stack. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    params : list
        params = [layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max, fill_value, shape, Xsize, Ysize, left_crop, verbose]
        layer_id : int
            Layer ID should be a value bewteen -1 and self.nz_tiles-1. -1 means the last layer will be assembled.
        fls_layer : list
            List of files for individual tiles.
        image_name : str
            Image name ('RawImageA' or 'RawImageB').
        tr_matr_layer : list
            List of transformation matrices for individual tiles.
        weight_min : float
            vmin for weight.
        weight_max : float
            vmax for weight.
        fill_value : int
            The value to assign to pixels outside the transformed image bounds.
        shape : list of two ints
            mosaic shape (ny_tiles, nx_tiles).
        Xsize : int
            Overall Mosaic width (pixels).
        Ysize : int
            Overall Mosaic height (pixels).
        left_crop : int 
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own).
        verbose : boolean
            Display intermediate results.
    deformation_field : 2D array
        Deformation field for distortion corrections to be executed. If is np.nan - no distortion correction.

    Returns:
    ----------
    layer_mosaic, layer_id
    '''
    layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max, fill_value, shape, Xsize, Ysize, left_crop, verbose = params
    layer_mosaic = np.zeros((Ysize, Xsize-left_crop), dtype=float)
    layer_mosaic_weights = np.zeros((Ysize, Xsize-left_crop), dtype=float)
    tile_params_mult = []
    xy_limits = []
    for fl, (j, tr_matr_single) in zip(tqdm(fls_layer, desc = 'Building tile parameter sets', display = verbose), enumerate(tr_matr_layer)):
        tile_params_mult.append([j, fl, image_name, tr_matr_single, Ysize, Xsize, weight_min, weight_max, left_crop])
    if len(tile_params_mult)>0:
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
        layer_mosaic_weights = np.clip(layer_mosaic_weights, weight_min, weight_max*np.prod(shape)) 
        layer_mosaic = np.nan_to_num(layer_mosaic / layer_mosaic_weights, nan=-fill_value)
    return layer_mosaic, layer_id


def generate_report_mill_rate_montage_xlsx(Mill_Rate_Data_xlsx, **kwargs):
    '''
    Generate Report Plot for mill rate evaluation from XLSX spreadsheet file. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    Mill_Rate_Data_xlsx : str
        Path to the XLSX spreadsheet file containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.
    
    kwargs:
    ----------
    Mill_Volt_Rate_um_per_V : float
        Milling Voltage to Z conversion (µm/V). Default is 31.235258870176065.
    mosaic_shape : tuple or list of of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of of 2 ints
        Y and X indices of the tile for which the WD fs frame will be evaluzted. Default is (0, 0).
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
        save_fname = kwargs.get ('save_fname', os.path.join(data_dir, Mill_Rate_Data_xlsx.replace('.xlsx','_Mill_Rate.png')))
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
    mosaic_shape : tuple or list of of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of of 2 ints
        Y and X indices of the tile for which the WD fs frame will be evaluzted. Default is (0, 0).
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
            my_col = plt.get_cmap("gist_rainbow_r")((nxny-l)/nxny)
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
            save_fname = kwargs.get ('save_fname', os.path.join(data_dir, FIBSEM_Data_xlsx.replace('.xlsx', '_' + SEM_params[k] + '.png')))
            axs[-1].text(-0.12, -0.23, save_fname, fontsize = 5, transform=axs[-1].transAxes)
            fig.savefig(save_fname, dpi=dpi)
        else:
            save_fname = 'Image not saved'
        save_fnames.append(save_fname) 
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
    mosaic_shape : tuple or list of of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    frame_id : int
        ID of the frame to show the SEM parameter map over the tile mosaic.
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
            my_col = plt.get_cmap("gist_rainbow_r")((ny-j)/(ny))
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
        save_fname = kwargs.get ('save_fname', os.path.join(data_dir, FIBSEM_Data_xlsx.replace('.xlsx',fname_repl_suffix+'_frame{:d}.png'.format(frame_id))))
        axs[-1].text(-0.12, -0.07, save_fname, fontsize = 4, transform=axs[-1].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    else:
        save_fname = 'Image not saved'
    return save_fname


def generate_report_data_minmax_montage_xlsx(minmax_xlsx_file, **kwargs):
    '''
    Generate Report Plot for data Min-Max from XLSX spreadsheet file. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    minmax_xlsx_file : str
        Path to the XLSX spreadsheet file containing Min-Max data

    kwargs:
    ----------
    mosaic_shape : tuple or list of of 2 ints
        Mosaic shape (ny_tiles, nx_tiles). Default is (1,1).
    tile_id : tuple or list of of 2 ints
        Y and X indices of the tile for which the WD fs frame will be evaluzted. Default is (0, 0).
    save_png : boolean
        If True (default), the plot is saved into PNG file.
    dpi : int
        DPI for PNG. Default is 300.
    save_fname : string
        File name to save the PNG image. Default is os.path.join(data_dir, minmax_xlsx_file.replace('.xlsx','_Min_Max.png')).
    verbose : boolean
        Display intermediate results. Default is False.

    Returns:
    ----------
    save_fname
    '''
    saved_kwargs = read_kwargs_xlsx(minmax_xlsx_file, 'kwargs Info', **kwargs)
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
        save_fname = kwargs.get ('save_fname', os.path.join(data_dir, minmax_xlsx_file.replace('.xlsx','_Min_Max.png')))
    else:
        save_fname = 'Image not saved'
    if verbose:
        print('Loading kwarg Data')
    Sample_ID = kwargs.get('Sample_ID', saved_kwargs.get('Sample_ID', ''))
    thr_min = saved_kwargs.get("thr_min", 0.0)
    thr_max = saved_kwargs.get("thr_max", 0.0)
    fit_params_saved = saved_kwargs.get("fit_params", ['SG', 101, 3])
    fit_params = kwargs.get("fit_params", fit_params_saved)
    preserve_scales =  saved_kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using the settings defined by fit_params below
    
    if verbose:
        print('Loading MinMax Data')
    try:
        int_results_all = pd.read_excel(minmax_xlsx_file, sheet_name='FIBSEM Data')
    except:
        int_results_all = pd.read_excel(minmax_xlsx_file, sheet_name='MinMax Data')
    
    int_results = int_results_all.iloc[mosaic_shape[0]*tile_id[0]+tile_id[1]::nxny, :]
    frames = int_results['Frame']/nxny
    frame_min = np.array(int_results['Min'])
    frame_max = np.array(int_results['Max'])
    data_min_glob  = np.min(frame_min)
    data_max_glob  = np.max(frame_max)

    if verbose:
        print('Generating Plots')
    fs = 12

    fig, axs = plt.subplots(3,1, figsize = (6,10), sharex=True)
    fig.subplots_adjust(left=0.15, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.05)
    
    for k in np.arange(nxny):
        my_col = plt.get_cmap("gist_rainbow_r")((nxny-k)/(nxny-1))
        framek_min = int_results_all.iloc[k::nxny, :]['Min']
        framek_max = int_results_all.iloc[k::nxny, :]['Max']
        if k == mosaic_shape[1]*tile_id[0]+tile_id[1]:
            axs[0].plot(frames, framek_min, color=my_col, marker='x', markersize=4)
            axs[1].plot(frames, framek_max, color=my_col, marker='x', markersize=4)
        else:
            axs[0].plot(frames, framek_min, color=my_col)
            axs[1].plot(frames, framek_max, color=my_col)
    axs[0].set_ylabel('All Tiles Minima Values')
    axs[1].set_ylabel('All Tiles Maxima Values')

    if fit_params[0] != 'None':
        sv_apert = min([fit_params[1], len(frames)//8*2+1])
        print('Using fit_params: ', 'SG', sv_apert, fit_params[2])
        sliding_min = savgol_filter(frame_min.astype(np.double), sv_apert, fit_params[2])
        sliding_max = savgol_filter(frame_max.astype(np.double), sv_apert, fit_params[2])
    else:
        print('Not smoothing the Min/Max data')
        sliding_min = frame_min.astype(np.double)
        sliding_max = frame_max.astype(np.double)

    axs[0].text(0.2, 1.04, Sample_ID, fontsize = fs, transform=axs[0].transAxes)
    axs[2].plot(frame_min, 'b', linewidth=1, label='Frame Minima')
    axs[2].plot(sliding_min, 'b', linewidth=2, linestyle = 'dotted', label='Sliding Minima')
    axs[2].plot(frame_max, 'r', linewidth=1, label='Frame Maxima')
    axs[2].plot(sliding_max, 'r', linewidth=2, linestyle = 'dotted', label='Sliding Maxima')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Tile ({:d},{:d}) Minima and Maxima Values'.format(*tile_id))
    dxn = (data_max_glob - data_min_glob)*0.1
    axs[2].set_ylim((data_min_glob - dxn, data_max_glob+dxn))
    xminmax = [0, len(frame_min)]
    y_min = [data_min_glob, data_min_glob]
    y_max = [data_max_glob, data_max_glob]
    axs[2].plot(xminmax, y_min, 'b', linestyle = '--')
    axs[2].plot(xminmax, y_max, 'r', linestyle = '--')
    axs[2].text(len(frame_min)/20.0, data_min_glob-dxn/1.75, 'data_min_glob={:.1f}'.format(data_min_glob), fontsize = fs-2, c='b')
    axs[2].text(len(frame_min)/20.0, data_max_glob+dxn/2.25, 'data_max_glob={:.1f}'.format(data_max_glob), fontsize = fs-2, c='r')
    axs[2].text(len(frame_min)/20.0, data_min_glob+dxn*4.5, 'thr_min={:.1e}'.format(thr_min), fontsize = fs-2, c='b')
    axs[2].text(len(frame_min)/20.0, data_min_glob+dxn*5.5, 'thr_max={:.1e}'.format(thr_max), fontsize = fs-2, c='r')
    for ax in axs:
        ax.grid(True)
    if save_png:
        axs[2].text(-0.12, -0.17, save_fname, fontsize = 5, transform=axs[2].transAxes)
        fig.savefig(save_fname, dpi=dpi)
    return save_fname


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
    index_pairs : array of pairs of absolute (in 1D sense of fls.ravel()) tile indices. Auto-determined during initialization, depends of grid setting.
        if grid == 'rect':  index_pairs = np.array(col_ind).reshape((row, 2))
    Sample_ID : str
            Sample ID
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    PixelSize : float
        pixel size in nm. This is inherited from FIBSEM_frame object. Default is 8.0
    voxel_size : rec.array(( float,  float,  float), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        voxel size in nm. Default is isotropic (PixelSize, PixelSize, PixelSize)
    Scaling : 2D array of floats
        scaling parameters allowing to convert I16 data into actual electron counts 
    fnm_reg : str
        filename for the final registered dataset
    use_DASK : boolean
        use python DASK package to parallelize the computation or not (False is used mostly for debug purposes).
    thr_min : float
        CDF threshold for determining the minimum data value
    thr_max : float
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
            XScaleShiftTransform  -  x-scale, x-shift, y-shift
            ScaleShiftTransform - x-scale, y-scale, x-shift, y-shift
            AffineTransform -  full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift)
            RegularizedAffineTransform - full Affine (x-scale, y-scale, rotation, shear, x-shift, y-shift) with regularization on deviation from ShiftTransform
    l2_matrix : 2D float array
        matrix of regularization (shrinkage) parameters
    targ_vector : 1D float array
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
    save_parameters(**kwargs):
        Save transformation attributes and parameters (including transformation matrices)

    evaluate_FIBSEM_statistics(**kwargs)
        Evaluates parameters of FIBSEM data set (data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).

    extract_keypoints(**kwargs):
        Extract Key-Points and Descriptors

    determine_transformations_SIFT(self, **kwargs)
        Determine transformation matrices for frame pairs using SIFT. 

    SIFT_evaluation(index_pair, pair_margins, **kwargs)
        Evaluate SIFT performance on a given index_pair.

    determine_transformations_ECC(**kwargs)
        Determine transformation matrices for frame pairs using ECC. Uses find_Transform_ECC(img1, img2, **kwargs).

    solve_stack_stitching(**kwargs)
        Solve mosaic stack stitching (perform bundle optimization).

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
        image_coordinates_file : str
            Path to a whitespace-delimited text file specifying tile coordinates,
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
        intralayer_weight : float, default 1.0
            Weight for pairwise constraints within a single Z-layer.
        interlayer_weight : float, default 100.0
            Weight for pairwise constraints for tiles between adjacent Z-layers.(100–10000 typical).
        add_reverse_edges : bool, default False
            If True, adds both (i->j) and (j->i) with same weight (increases robustness).
        shape : tuple of two int (self.ny_tiles, self.nx_tiles)
            The program will try to auto-determine the shape, but it can be set explicitly.
                # self.ny_tiles  - # of rows per layer (# of tiles along Y-axis)
                # self.nx_tiles  - # of columns per layer(# of tiles along X-axis)
        EightBit : int
            If 1 then the data is assumed uint8, otherwise int16
        U8_conversion : str
            Range selection for U8 conversion. Options are: 'global', 'sliding', and 'local'. Default is 'local'.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3.
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
        targ_vector : 1D float array
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
        perform_deformation : boolean
            If True - the data is deformed (in addition to transformation defined above) using the deformation field data defined below.
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
        image_coordinates_file = kwargs.get('image_coordinates_file', '')
        metadata_file = kwargs.get('metadata_file', '')
        self.data_dir = kwargs.get('data_dir', os.path.split(self.fls.ravel()[0])[0])
        self.ftype = kwargs.get('ftype', 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files, ftype=2 for PNG files
        self.intralayer_weight = kwargs.get('intralayer_weight', 1.0)
        self.interlayer_weight = kwargs.get('interlayer_weight', 100.0)
        self.add_reverse_edges = kwargs.get('add_reverse_edges', False)
        self.U8_conversion = kwargs.get('U8_conversion', 'local')
        if self.ftype == 0:
            test_frame = FIBSEM_frame(self.fls.ravel()[0], ftype = self.ftype, calculate_scaled_images=False, read_header_only=True)
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
            self.DetA = test_frame.DetA
            self.DetB = test_frame.DetB
            self.Notes = test_frame.Notes
            self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
            if self.DetB == 'None':
                ImgB_fraction = 0.0
            self.BrightnessA = test_frame.BrightnessA 
            self.BrightnessB = test_frame.BrightnessB
            self.ContrastA = test_frame.ContrastA
            self.ContrastB = test_frame.ContrastB
            self.Sample_ID = kwargs.get("Sample_ID", test_frame.Sample_ID)
            self.EightBit = kwargs.get("EightBit", test_frame.EightBit)
        if self.ftype ==2:
            metadata_file_default = os.path.join(os.path.split(self.fls.ravel()[0])[0], 'metadata.txt')
        if os.path.exists(metadata_file_default):
            metadata_file = kwargs.get('metadata_file', metadata_file_default)
        else:
            metadata_file = kwargs.get('metadata_file', '')
        self.metadata_file = metadata_file
        if self.ftype == 2 and metadata_file:
            metadata = parse_metadata_file(metadata_file)
            '''
            Operator:           Templier
            Experiment:         wafer_53_scan_008
            Time:               5/9/2022 6:25:27 AM
            FoVX:               16.000µm
            FoVY:               13.984µm
            Width:              2000px
            Height:             1748px
            Pixelsize:          8.000nm
            Thumbnail-Scale:    0.25
            Scanspeed:          4
            Dwelltime:          400ns
            Landing Energy:     1.200keV
            Beam Current:       552pA
            Stage pos. X:       -133.047µm
            Stage pos. Y:       -23813.574µm
            Stage pos. Z:       39405.312µm
            Stigmator Shift X:  0.000
            Stigmator Shift Y:  0.000
            Stigmator Tilt X:   1.615
            Stigmator Tilt Y:   0.405
            Projective X:       -536.562
            Projective Y:       333.766
            Lens:               1910436.229
            Focus Offset:       4.965µm
            Overall time:       201.043s
            Stage pos. X target:        -133.047µm
            Stage pos. Y target:        -23813.574µm
            Stage pos. Z target:        39405.312µm
            '''
            test_frame = FIBSEM_frame(self.fls.ravel()[0], ftype = 2)
            self.metadata = metadata
            ys, xs = test_frame.RawImageA.shape
            self.ScanRate = kwargs.get('PixelSize', 1e9/metadata.get('Dwelltime_ns', 100.0))
            self.EHT = kwargs.get('EHT', metadata.get('Landing_Energy_keV', 0))
            self.SEMCurr = kwargs.get('SEMCurr', metadata.get('Beam_Current_pA', 0.0)/1e12)
            self.XResolution = kwargs.get('XResolution', metadata.get('Width'), xs)
            self.YResolution = kwargs.get("YResolution", metadata.get('Height'), ys)
            self.XResolutions = kwargs.get('XResolutions', np.full(len(fls[0]), self.XResolution))
            self.YResolutions = kwargs.get('YResolutions', np.full(len(fls[0]), self.YResolution))
            self.PixelSize = kwargs.get('PixelSize', metadata.get('Pixelsize_nm', 5.0))
            self.EightBit = kwargs.get('EightBit', 1)
            self.Sample_ID = kwargs.get("Sample_ID",  metadata.get('Experiment', ''))
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
            fnm_mosaic_stack_default = os.path.splitext(os.path.split(self.fls.ravel()[0])[1])[0][0:-5] + 'mosaic_stack.mrc'
        except:
            fnm_mosaic_stack_default = 'mosaic_stack.mrc'
        self.fnm_mosaic_stack = kwargs.get('fnm_mosaic_stack', fnm_mosaic_stack_default)
        self.dtp = kwargs.get("dtp", np.int16)
        self.nz_tiles = self.fls.shape[0]
        self.n_tiles_per_layer = len(self.fls[0].ravel())
        kwargs.update({'data_dir' : self.data_dir, 'fnm_mosaic_stack' : self.fnm_mosaic_stack, 'dtp' : self.dtp})

        w_sqrt_intra = np.sqrt(self.intralayer_weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||
        w_sqrt_inter = np.sqrt(self.interlayer_weight)

        if image_coordinates_file: # suser-defined grid with FirstPixels determined from the image_coordinates_file file
            coord_dict = {}
            with open(image_coordinates_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        coord_dict[parts[0]] = (float(parts[1]), float(parts[2]))
            self.FirstPixels = np.array(FirstPixels)
            # Find all intra-layer neighbouring pairs by proximity.
            # Two tiles are neighbours if their bounding boxes overlap in both X and Y.
            # This naturally handles hexagonal layouts where each tile has 1 left/right
            # neighbour and up to 2 top/bottom neighbours.

            intra_index_pairs_x = []
            intra_margins_x = []
            intra_index_pairs_y = []
            intra_margins_y = []
            for i in range(self.n_tiles_per_layer):
                for j in range(i + 1, self.n_tiles_per_layer):
                    dx = abs(self.FirstPixels[j, 0] - self.FirstPixels[i, 0])
                    dy = abs(self.FirstPixels[j, 1] - self.FirstPixels[i, 1])
                    x_overlap = self.XResolution - dx
                    y_overlap = self.YResolution - dy
                    if x_overlap > 0 and y_overlap > 0:
                        ymargin = min(self.YResolution, max(100, int(2 * y_overlap)))
                        xmargin = min(self.XResolution, max(100, int(2 * x_overlap)))
                        if xmargin < ymargin:
                            intra_index_pairs_x.append((i, j))
                            intra_margins_x.append([ymargin, xmargin])
                        else:
                            intra_index_pairs_y.append((i, j))
                            intra_margins_y.append([ymargin, xmargin])
            L = self.nz_tiles                
            nh = L * len(intra_index_pairs_x)              # Total number of left-right intra-layer pairs
            nv = L * len(intra_index_pairs_y)              # Total number of up-down intra-layer pairs
            intra_index_pairs = np.array(intra_index_pairs_x + intra_index_pairs_y)
            intra_margins = intra_margins_x + intra_margins_y
            n_intra_single_layer = len(intra_index_pairs)
            n_intra = L * n_intra_single_layer
            nl = (L - 1) * self.n_tiles_per_layer
            C = n_intra + nl
            V = L * self.n_tiles_per_layer                     # Total number of tiles

            if verbose:
                print('Total number of tiles: ', V)
                print('Total number of left-right intra-layer pairs: ', nh)
                print('Total number of up-down intra-layer pairs: ', nv)
                print('Total number of intra-layer pairs: ', n_intra)
                print('Total number of inter-layer pairs: ', nl)
                print('Total number of of pairs (pair-wise translations): ', C)

            # Prepare data for sparse matrix A
            data = []
            row_ind = []
            col_ind = []
            row = 0   # row (entry) in the sparse matrix A (not a tile row)

            # Build a sparse matrix A for Ax=b lsqr equation
            # idx1 and idx2 are absolute (in 1D sense) tile indices
            # each entry is a single sparse matrix element, there are two elements per pairwise translation condition, they enter with opposite signs

            # Intra-layer adjacent pairs
            for l in range(L):
                for i in range(n_intra_single_layer):    
                    idx1 = l * self.n_tiles_per_layer + intra_index_pairs[i, 0]
                    idx2 = l * self.n_tiles_per_layer + intra_index_pairs[i, 1]
                    row_ind.extend([row, row])
                    col_ind.extend([idx1, idx2])
                    data.extend([-w_sqrt_intra, w_sqrt_intra])
                    row += 1

            # Inter-layer adjacent pairs
            for l in range(L - 1):
                for i in range(self.n_tiles_per_layer):
                    idx1 = l * self.n_tiles_per_layer + i
                    idx2 = (l + 1) * self.n_tiles_per_layer + i
                    row_ind.extend([row, row])
                    col_ind.extend([idx1, idx2])
                    data.extend([-w_sqrt_inter, w_sqrt_inter])
                    row += 1

        else:   # standard recti-linear grid with FirstPixels determined from the headers of .dat files
            FirstPixels = []
            for fl in fls[0].ravel():
                fr = FIBSEM_frame(fl, read_header_only=True)
                FirstPixels.append([fr.FirstPixelX, fr.FirstPixelY])
            self.FirstPixels = np.array(FirstPixels)
    
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

            # create the structure for pairwice tile transformation
            L = self.nz_tiles
            M = self.ny_tiles
            N = self.nx_tiles
            V = L * M * N                     # Total number of tiles
            nh = L * M * (N - 1)              # Total number of left-right intra-layer pairs
            nv = L * (M - 1) * N              # Total number of up-down intra-layer pairs
            n_intra = nh + nv
            nl = (L - 1) * M * N              # Total number of inter-layer pairs
            C = nh + nv + nl                  # Total number of of pairs (pair-wise translations)
            if verbose:
                print('Total number of tiles: ', V)
                print('Total number of left-right intra-layer pairs: ', nh)
                print('Total number of up-down intra-layer pairs: ', nv)
                print('Total number of intra-layer pairs: ', n_intra)
                print('Total number of inter-layer pairs: ', nl)
                print('Total number of of pairs (pair-wise translations): ', C)

            # Prepare data for sparse matrix A
            data = []
            row_ind = []
            col_ind = []
            row = 0   # row (entry) in the sparse matrix A (not a tile row)

            # Build a sparse matrix A for Ax=b lsqr equation
            # idx1 and idx2 are absolute (in 1D sense) tile indices
            # each entry is a single sparse matrix element, there are two elements per pairwise translation condition, they enter with opposite signs

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

        self.index_pairs = np.array(col_ind).reshape((row, 2))   # absolute (in 1D sense) tile indices for each pair
        self.Xoverlap = self.XResolution - (self.FirstPixels[1, 0] - self.FirstPixels[0, 0])
        i1, i2 = self.index_pairs[nh, :]
        self.Yoverlap = self.YResolution - np.abs((self.FirstPixels[i1, 1] - self.FirstPixels[i2, 1]))
        self.pair_margins = [[self.YResolution, 2*self.Xoverlap] for x in np.arange(nh)] + [[2*self.Yoverlap, self.XResolution] for x in np.arange(nv)] + [[self.YResolution, self.XResolution] for x in np.arange(nl)]
        self.A_csr = csr_matrix((data, (row_ind, col_ind)), shape=(C, V)) # sparse matrix

        eye3x3 = np.eye(3,3)
        self.ECC_transformation_matrices = np.repeat(eye3x3[np.newaxis, :, :], C, axis=0)
        self.ECC_transformation_valid = np.full(C, False)
        self.SIFT_transformation_matrices = np.repeat(eye3x3[np.newaxis, :, :], C, axis=0)
        self.SIFT_transformation_valid = np.full(C, False)
        self.SIFT_fnms_matches = ['' for x in np.arange(C)]
        self.SIFT_nmatches = np.full(C, 0)

        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Initialized FIBSEM_mosaic_dataset instance:')
            print('Number of tiles per Z-layer: {:d}'.format(self.n_tiles_per_layer))
            print('Total number of tile files: {:d}'.format(V))
            print('Number of Z-slices (nz_tiles): {:d}'.format(self.nz_tiles))
            print('Total number of pairwise transformations : {:d}'.format(C))
            #print('Index of the top-left pair in the last Z-layer: ', (L -1)* M * (N - 1))
    
        if image_coordinates_file:
            self.Xsize = np.max(self.FirstPixels[:, 0]) - np.min(self.FirstPixels[0, 0]) + self.XResolution
            self.Ysize = np.max(self.FirstPixels[:, 1]) - np.min(self.FirstPixels[0, 1]) + self.YResolution
        else:
            # initialize the montage size (assuming rectangular shape)
            self.Xsize = self.shape[1] * (self.XResolution - self.Xoverlap) + self.Xoverlap
            self.Ysize = self.shape[0] * (self.YResolution - self.Yoverlap) + self.Yoverlap
        
        # initialize the translation matrix for each tile
        shifts_x = self.FirstPixels[:, 0] - self.FirstPixels[0, 0]
        shifts_y = self.FirstPixels[:, 1] - self.FirstPixels[0, 1]
        single_layer_tr_matr = np.repeat(eye3x3[np.newaxis, :, :], self.n_tiles_per_layer, axis=0)
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
        if kwargs.get("recall_parameters", False):
            dump_filename = kwargs.get("dump_filename", '')
            try:
                dump_data = pickle.load(open(dump_filename, 'rb'))
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Loaded the data from the dump filename: ', dump_filename)
                dump_loaded = True
            except Exception as ex1:
                dump_loaded = False
                if verbose:
                    print('')
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Failed to open Parameter dump filename: ', dump_filename)
                    print(str(ex1))
            if dump_loaded:
                try:
                    for key in tqdm(dump_data, desc='Recalling the data set parameters'):
                        setattr(self, key, dump_data[key])
                except Exception as ex2:
                    if verbose:
                        print('')
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Failed to restore the object parameters from dump filename: ', dump_filename)
                        print(str(ex2))


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
        FIBSEM_Data_xlsx : str
            File path of the Excell file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).
        use_existing_data : boolean
            Default is False. If True and the data exists (saved into XLSX), use that.            
        verbose : boolean
            If True, intermediate messages and results will be displayed. Default is False.

        Returns:
        ----------
        FIBSEM_Data : list of 20 parameters
            FIBSEM_Data_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions, SEMStiX, SEMStiY, SEMAlnX, SEMAlnY, errors_s2
                FIBSEM_Data_xlsx : str
                    path to Excel file with the FIBSEM data
                data_min_glob : float   
                    min data value for I8 conversion (open CV SIFT requires I8)
                data_max_glob : float   
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
        verbose = kwargs.get('verbose', True)
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = True)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        frame_inds = kwargs.get("frame_inds", np.arange(self.nz_tiles))
        data_dir = kwargs.get('data_dir', self.data_dir)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        fit_params = kwargs.get('fit_params', ['SG', 3, 1])
        FIBSEM_Data_xlsx_default = os.path.join(data_dir, os.path.splitext(self.fnm_mosaic_stack)[0] + '_FIBSEM_Data.xlsx')
        FIBSEM_Data_xlsx = kwargs.get('FIBSEM_Data_xlsx', FIBSEM_Data_xlsx_default)
        use_existing_data = kwargs.get('use_existing_data', False)

        if hasattr(self, 'Mill_Volt_Rate_um_per_V'):
            Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", self.Mill_Volt_Rate_um_per_V)
        else:
            Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", 31.235258870176065)

        local_kwargs = {'use_DASK' : use_DASK,
                        'DASK_client_retries' : DASK_client_retries,
                        'ftype' : ftype,
                        'frame_inds' : np.arange(len(self.fls.ravel())),
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
        WD = self.FIBSEM_Data[5]
        
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
        
        MillingYVoltage = self.FIBSEM_Data[6]
        frame_inds_ext = np.repeat(np.array(frame_inds), self.nx_tiles*self.ny_tiles)

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

            self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  Z_pixel_size_WD), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        except:
            if verbose:
                print('Could not estimate milling rate')
            self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        if verbose:
            print('Set the voxel size to: ', self.voxel_size)

        return self.FIBSEM_Data
    

    def extract_keypoints(self, **kwargs):
        '''
        Extract Key-Points and Descriptors. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        DASK_client : DASK client. If empty string '' (Default), local computations are performed.
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
            Deformation field for distortion corrections to be executed before SIFT. Default is np.nan - no distortion correction.
            Deformation field should be passed as shared_data = shared_data_future since it is the same for all tiles.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        U8_conversion : str
            Range selection for U8 conversion. Options are: 'global', 'sliding', and 'local'. Default is 'local'.
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
        use_DASK, status_update_address = check_DASK(DASK_client, verbose = verbose)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        if hasattr(self, 'U8_conversion'):
            U8_conversion = kwargs.get('U8_conversion', self.U8_conversion)
        else:
            U8_conversion = kwargs.get('U8_conversion', 'local')
        if U8_conversion != 'local':
            data_minmax = kwargs.get("data_minmax", self.data_minmax)
            minmax_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding = data_minmax
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        deformation_field = kwargs.get('deformation_field', np.nan)
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
                    'fill_value' : fill_value}

        if U8_conversion == 'sliding':
            params_s3 = []
            for j, fl in enumerate(self.fls.ravel()):
                params_s3. append([fl, data_min_sliding[j], data_max_sliding[j], kpt_kwargs])
        else:
            if U8_conversion == 'global': 
                params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls.ravel()]
            else:
                params_s3 = [[fl, -1, -1, kpt_kwargs] for fl in self.fls.ravel()]
  
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
        Determine transformation matrices for frame pairs using SIFT. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        pair_margins : array of tuples of 2 ints
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
        targ_vector : 1D float array
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
                print('No data on individual key-point data files, perform key-point search')
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
            if hasattr(self, 'SIFT_nmatches_min'):
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

            for index_pair, pair_margins  in zip(tqdm(self.index_pairs, desc='Setting up SIFT parameter list', display=verbose), self.pair_margins):
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
            if nl > 0:
                print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Mean Number of Matched Keypoints for inter-layer matches :', np.mean(self.SIFT_nmatches[nh+nv:]).astype(np.int64))
            print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   {:d} out of {:d} SIFT transformations are valid  (SIFT_nmatches > {:d})'.format(np.sum(self.SIFT_transformation_valid), C, SIFT_nmatches_min))
        return transformations_results_3D


    def SIFT_evaluation(self, index_pair, pair_margins, **kwargs):
        '''
        Evaluate SIFT performance on a given index_pair. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        Parameters:
        index_pair : tuple of 2 ints
            Pair of absolute (in 1D sense of fls.ravel()) tile indices.
        pair_margins : tuples of 2 ints
            Parts of images to be used. It is assumed that first image (img1) in each target_pair is to the left and above of the second image (img2).
            Subsets img1[-ymargin:, :] and  img2[0:ymargin, :] or img1[:, -xmargin:] and  img2[:, 0:xmargin] will be used for correlation.
            Default is full images, so image_margins = (self.YResolution, self.XResolution)

        kwargs:
        ----------
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        left_crop : int
            Cropping value for cropping the image from the left side (used along with deformation_filed or on its own). Default is 0 - no cropping.
        deformation_field : 3D array
            Deformation field for distortion corrections to be executed before SIFT. Default is np.nan - no distortion correction
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
            A path for saving PNG data. Default is auto-generated as os.path.join(self.data_dir, os.path.split(fnm_matches)[1].replace('_matches.bin') + '_SIFT_test.png').
        Returns:
        ----------
        fnm_deformed1, fnm_deformed2, transformations_result, int_results
            int_results is pd.Dataframe with columns: ['X-src', 'Y-src', 'X-src transformed', 'Y-src transformed', 'X-dst', 'Y-dst', 'X-error', 'Y-error', 'Int-src', 'Int-dst']
        '''
        ftype = kwargs.get("ftype", self.ftype)
        left_crop = kwargs.get('left_crop', 0)
        deformation_field = kwargs.get('deformation_field', np.nan)
        perform_deformation = np.any(np.invert(np.isnan(deformation_field)))
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
        if hasattr(self, 'SIFT_nmatches_min'):
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', self.SIFT_nmatches_min)
        else:
            SIFT_nmatches_min = kwargs.get('SIFT_nmatches_min', 5)
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
        fnms_kpts = []
        for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ', display=verbose)):
            fnms_kpts.append(extract_keypoints_descr_files(param_s3, deformation_field))
        kpp1s, des1, kpt_int1 = pickle.load(open(fnms_kpts[0], 'rb'))
        n_kpts1 = len(kpp1s)
        kpp2s, des2, kpt_int2 = pickle.load(open(fnms_kpts[1], 'rb'))
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
        index_loc0, index_loc1 = np.mod(index_pair, self.nx_tiles*self.ny_tiles)
        FirstPixels_delta = self.FirstPixels[index_loc1] - self.FirstPixels[index_loc0]
        ymargin, xmargin = pair_margins
        dt_kwargs['warp_matrix'] = np.array([[1, 0, -FirstPixels_delta[0]], [0, 1, -FirstPixels_delta[1]]], dtype=np.float32)
        dt_kwargs['image_margins'] = (ymargin, xmargin)
        dt_kwargs['image_shape'] = (self.YResolution, self.XResolution)
        dt_kwargs['left_crop'] = left_crop
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
                #axx.plot([xbins[indxi]+dbx, xbins[indxa]+dbx], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
                axx.plot([xbins[indxi], xbins[indxa]], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
                axx.plot([xbins[mxx_ind]+dbx], [mxx], 'rd')
                axx.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(xshifts)), transform=axx.transAxes, fontsize=fsz)
                axx.text(0.05, 0.8, 'median={:.3f}'.format(np.median(xshifts)), transform=axx.transAxes, fontsize=fsz)
                axx.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMx), transform=axx.transAxes, fontsize=fsz)
                ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64)
                error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, estimation=estimation, start=start, max_aver_aperture=5)
                dby = (ybins[1]-ybins[0])/2.0
                #axy.plot([ybins[indyi] + dby, ybins[indya] + dby], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
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
            axs[0].text(0.01, 1.00 - 0.175*frame.XResolution/frame.YResolution, 'Image Margins: {:d}, {:d}'.format(*pair_margins), fontsize=fsize_text, transform=axs[0].transAxes)

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
                axs[0].text(0.01, 1.00 - 0.215*frame.XResolution/frame.YResolution, 'mean_error = {:.3f}, error_FWHMx = {:3f},  error_FWHMy={:3f}'.format(error_abs_mean, error_FWHMx, error_FWHMy), fontsize=fsize_text, transform=axs[0].transAxes) 

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
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        pair_margins : array of tuples of 2 ints
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
        ----------
        transformations_results_3D : array of lists containing the results:
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

        for index_pair, pair_margins  in zip(tqdm(self.index_pairs, desc='Setting up ECC parameter list', display=verbose), self.pair_margins):
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
        Solve mosaic stack stitching (perform bundle optimization). ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        verbose : boolean
            Display intermediate results. Default is True.
        method : string
            Options are: ['SIFT-ECC', 'SIFT', 'ECC']. Default is 'ECC'.  'SIFT-ECC' means - try SIFT first, and for the tiles that SIFT failed, try ECC.
        
        Returns:
        ----------
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
        C = nh + nv + nl                  # Total number of pairs (pair-wise translations)
        # horiz_trans: np.ndarray (L, M, N-1, 2), translations to right neighbor (x,y)
        # vert_trans: np.ndarray (L, M-1, N, 2), translations to bottom neighbor (x,y)
        # layer_trans: np.ndarray (L-1, M, N, 2), translations to upper layer (x,y)

        # We already have self.A_csr = csr_matrix((data, (row_ind, col_ind)), shape=(C, V)) # sparse matrix
        # Now we need to contruct the matrix B and solve LSQ

        w_sqrt_intra = np.sqrt(self.intralayer_weight)  # because LSQR minimizes ||W^{1/2} (Ax - b)||
        w_sqrt_inter = np.sqrt(self.interlayer_weight)
        weights = np.concatenate((np.full((nh+nv), w_sqrt_intra), np.full(nl, w_sqrt_inter)))

        data = []
        row_ind = []
        col_ind = []
        row = 0   # row (entry) in the sparse matrix A (not a tile row)

        # Build a sparse matrix A for Ax=b lsqr equation
        # idx1 and idx2 are absolute (in 1D sense) tile indices
        # each entry is a single sparse matrix element, there are two elements per pairwise translation condition, they enter with opposite signs

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

        if method not in valid_methods:
            if verbose:
                print('Method ' + method +' is not among valid methods: ', valid_methods)
            return np.nan
        else:
            if method == 'SIFT':
                self.SIFT_residual_error_x = np.full(C, np.nan)
                self.SIFT_residual_error_y = np.full(C, np.nan)
                bx = self.SIFT_transformation_matrices[:, 0, 2] * weights
                by = self.SIFT_transformation_matrices[:, 1, 2] * weights
                res_x_all = lsqr(self.A_csr[self.SIFT_transformation_valid], bx[self.SIFT_transformation_valid])
                res_y_all = lsqr(self.A_csr[self.SIFT_transformation_valid], by[self.SIFT_transformation_valid])
                # calculate weighted residuals: b_weighted - A_weighted x
                self.SIFT_residual_error_x[self.SIFT_transformation_valid] = bx[self.SIFT_transformation_valid] - self.A_csr[self.SIFT_transformation_valid] @ res_x_all[0]
                self.SIFT_residual_error_y[self.SIFT_transformation_valid] = by[self.SIFT_transformation_valid] - self.A_csr[self.SIFT_transformation_valid] @ res_y_all[0]
                self.SIFT_r2norm_x = res_x_all[4]
                self.SIFT_r2norm_y = res_y_all[4]
            else:
                self.ECC_residual_error_x = np.full(C, np.nan)
                self.ECC_residual_error_y = np.full(C, np.nan)
                bx = self.ECC_transformation_matrices[:, 0, 2] * weights
                by = self.ECC_transformation_matrices[:, 1, 2] * weights
                res_x_all = lsqr(self.A_csr[self.ECC_transformation_valid], bx[self.ECC_transformation_valid])
                res_y_all = lsqr(self.A_csr[self.ECC_transformation_valid], by[self.ECC_transformation_valid])
                self.ECC_residual_error_x[self.ECC_transformation_valid] = bx[self.ECC_transformation_valid] - self.A_csr[self.ECC_transformation_valid] @ res_x_all[0]
                self.ECC_residual_error_y[self.ECC_transformation_valid] = by[self.ECC_transformation_valid] - self.A_csr[self.ECC_transformation_valid] @ res_y_all[0]
                self.ECC_r2norm_x = res_x_all[4]
                self.ECC_r2norm_y = res_y_all[4]
        res_x = res_x_all[0]
        res_y = res_y_all[0]
        positions = np.zeros((V, 2))
        positions[:, 0] = res_x - res_x[0]
        positions[:, 1] = res_y - res_y[0]
        self.tr_matr[:, :, 0:2, 2] = positions.reshape((L, M*N, 2))
        self.tile_positions = -positions.reshape((L, M*N, 2))

        return self.tile_positions


    def generate_transformation_report(self, **kwargs):
        '''
        Generate Report Plot for transformation summary. ©G.Shtengel 12/2022 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
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
        mosaic_shape = kwargs.get('mosaic_shape', (self.ny_tiles, self.nx_tiles))
        nxny = np.prod(mosaic_shape)
        tile_id = kwargs.get('tile_id', (0, 0))
        verbose = kwargs.get('verbose', False)
        save_png = kwargs.get('save_png', True)
        dpi = kwargs.get('dpi', 300)
        data_dir = kwargs.get('data_dir', self.data_dir)
        if save_png:
            try:
                save_fname = kwargs.get ('save_fname', os.path.splitext(self.fnm_mosaic_stack)[0] + '_Relative_Tile_Shifts.png')
            except:
                save_fname = kwargs.get ('save_fname', os.path.join(data_dir, 'Relative_Tile_Shifts.png'))
        else:
            save_fname = 'Image not saved'
        Sample_ID = kwargs.get('Sample_ID', self.Sample_ID)
        frame_inds = kwargs.get('frame_inds', np.arange(self.nz_tiles))
        tile_positions_x = self.tile_positions[frame_inds, :, 0] - self.tile_positions[0, :, 0]
        tile_positions_y = self.tile_positions[frame_inds, :, 1] - self.tile_positions[0, :, 1]

        if verbose:
            print('Generating Plot')
        fig, axs = plt.subplots(3,1, figsize = (6,10), sharex=True)
        fig.subplots_adjust(left=0.12, bottom=0.06, right=0.99, top=0.97, wspace=0.05, hspace=0.03)

        for k in np.arange(nxny):
            my_col = plt.get_cmap("gist_rainbow_r")((nxny-k)/(nxny-1))
            tile_positions_xk = tile_positions_x[:, k]
            tile_positions_yk = tile_positions_y[:, k]
            if k == self.nx_tiles*tile_id[0]+tile_id[1]:
                axs[0].plot(frame_inds, tile_positions_xk, color=my_col, marker='x', markersize=4, label='Tile ({:d},{:d}), X-shift'.format(*tile_id))
                axs[1].plot(frame_inds, tile_positions_yk, color=my_col, marker='x', markersize=4, label='Tile ({:d},{:d}), Y-shift'.format(*tile_id))
                axs[2].plot(frame_inds, tile_positions_xk, color='red', label='Tile ({:d},{:d}), X-shift'.format(*tile_id))
                axs[2].plot(frame_inds, tile_positions_yk, color='blue', label='Tile ({:d},{:d}), Y-shift'.format(*tile_id))
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
        verbose : boolean
            Display intermediate results. Default is False.

        Returns:
        ----------
        layer_mosaics, layer_id, layer_mosaic_weights, xy_limits
        
        '''
        ifDetB = (self.DetB != 'None')
        image_names = ['RawImageA']
        if ifDetB:
            image_names.append('RawImageB')
        if layer_id<-1 or layer_id>self.nz_tiles-1:
            print('layer_id parameter {:d} is out of range: -1 to {:d}'.format(layer_id, self.nz_tiles))
            return np.nan
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=True)
        deformation_field = kwargs.get('deformation_field', np.nan)
        left_crop = kwargs.get('left_crop', 0)
        weight_min = kwargs.get('weight_min', 1.0)
        weight_max = kwargs.get('weight_max', 2048.0) 
        fill_value = kwargs.get ('fill_value', -10000) 
        verbose = kwargs.get('verbose', False)
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
        dpi = kwargs.get('dpi', 300)

        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3) 

        layer_mosaics = []
        layer_mosaic_weights = np.zeros((self.Ysize, self.Xsize-left_crop), dtype=float)
        for image_name in image_names:
            layer_mosaic = np.zeros((self.Ysize, self.Xsize-left_crop), dtype=float)
            tile_params_mult = []
            xy_limits = []
            for fl, (j, tr_matr_single) in zip(tqdm(self.fls[layer_id].ravel(), desc = 'Building tile parameter sets', display = verbose), enumerate(self.tr_matr[layer_id])):
                tile_params_mult.append([j, fl, image_name, tr_matr_single, self.Ysize, self.Xsize, weight_min, weight_max, left_crop])
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
                layer_mosaic_weights = np.clip(layer_mosaic_weights, weight_min, weight_max*np.prod(self.shape)) 
                layer_mosaic = np.nan_to_num(layer_mosaic / layer_mosaic_weights, nan=-fill_value)
                layer_mosaics.append(layer_mosaic)

        if save_snapshot:
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
                                tile_positions = self.tile_positions[layer_id],
                                 linewidth=linewidth,
                                 linestyle=linestyle,
                                 edgecolor=color)
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
            fig.savefig(snapshot_fname, dpi=dpi)
        
        if save_to_dat:
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
            #ChanNum_new = 1
            #ChanNum_new_string =  pack('b', ChanNum_new)
            #header_new[32:33] = ChanNum_new_string
            #AI2_new = 0
            #AI2_new_string =  pack('b', AI2_new)
            #header_new[152:153] = AI2_new_string
            '''
            FirstPixelX_new_string =  pack('>l', FirstPixelX_new)
            header_new[70:74] = FirstPixelX_new_string
            FirstPixelY_new_string =  pack('>l', FirstPixelY_new)
            header_new[74:78] = FirstPixelY_new_string
            '''
            # Create new Raw data array
            dt = np.dtype(np.int16).newbyteorder('>')

            # Save new frame
            with open(dat_fname, 'wb') as f:
                f.write(header_new)
                for layer_mosaic in layer_mosaics:
                    layer_mosaic.reshape(-1).astype(dt).tofile(f)

        if save_images:
            imf1, imf2 = os.path.splitext(image_fname)
            for j, layer_mosaic in enumerate(layer_mosaics):
                sx = 15.0
                sy = sx / layer_mosaic.shape[1] * layer_mosaic.shape[0]
                fig, ax = plt.subplots(1,1, figsize=(sx,sy))
                fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
                if j == 0:
                    if not save_snapshot:
                        vmin, vmax = get_min_max_thresholds(layer_mosaic, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
                    else:
                        vmin = vminA
                        vmax = vmaxA
                    det_str = 'Detector A:  '+ self.DetA.strip('\x00')
                else:
                    if not save_snapshot:
                        vmin, vmax = get_min_max_thresholds(layer_mosaic, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
                    else:
                        vmin = vminB
                        vmax = vmaxB
                    det_str = 'Detector B:  '+ self.DetB.strip('\x00')
                print(det_str + ', data range: vmin={:.2f}, vmax={:.2f}'.format(vmin, vmax))
                ax.imshow(layer_mosaic, cmap='Greys', vmin = vmin, vmax = vmax)
                ax.axis(False)
                overlay_montage_grid(ax, self,
                                     tile_positions = self.tile_positions[layer_id],
                                     left_crop = left_crop,
                                     tile_positions_actual = True,
                                     linewidth=0.1, color = 'red')
                if j == 1:
                    image_fname_loc = imf1 + '_' + self.DetB.strip('\x00') + imf2
                else:
                    image_fname_loc = imf1 + '_' + self.DetA.strip('\x00') + imf2
                fig.savefig(image_fname_loc, dpi=dpi)

        return layer_mosaics, layer_id, layer_mosaic_weights, xy_limits


    def save_stack(self, **kwargs):
        '''
        Assemble all layers based on transformation matrices for each tile and save them into stack. ©G.Shtengel 01/2026 gleb.shtengel@gmail.com
        
        kwargs:
        ----------
        fnm_mosaic_stack : string
            Filename to save the data. Default is object attribute self.fnm_mosaic_stack
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['mrc', 'h5'].
            Defauls is ['mrc']. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        voxel_size : rec array of 3 elements
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
        image_name : str
            Image name ('RawImageA' or 'RawImageB'). Default is 'RawImageA'.
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int (default to 3)
            Number of allowed automatic retries if a task fails. Default is object attribute.
        verbose : boolean
            Display intermediate results. Default is False.

        Returns:
        ----------
        fnms_saved
        
        '''
        DASK_client = kwargs.get('DASK_client', '')
        fnm_mosaic_stack = kwargs.get('fnm_mosaic_stack', self.fnm_mosaic_stack)
        fnm_types = kwargs.get("fnm_types", ['mrc'])
        image_name = kwargs.get('image_name', 'RawImageA')
        if hasattr(self, 'voxel_size'):
            voxel_size = kwargs.get("voxel_size", self.voxel_size)
        else:
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
        use_DASK, status_update_address = check_DASK(DASK_client, verbose=True)
        deformation_field = kwargs.get('deformation_field', np.nan)
        DF0 = convert_tr_matr_into_deformation_field(np.eye(3,3).astype(float), (self.YResolution, self.XResolution))
        kwargs['deformation_field'] = deformation_field - DF0
        left_crop = kwargs.get('left_crop', 0)
        kwargs['left_crop'] = left_crop
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)

        fnms_saved = []
        if 'mrc' in fnm_types:
            mrc_filename = os.path.splitext(fnm_mosaic_stack)[0] + '.mrc'
            fnms_saved.append(mrc_filename)
            stack_shape = (self.nz_tiles, self.Ysize, self.Xsize-left_crop)
            mrc_new = mrcfile.new_mmap(mrc_filename, shape = stack_shape, mrc_mode=mrc_mode, overwrite=True)
            mrc_new.voxel_size = voxel_size_angstr
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the registered stack into the file: ', mrc_filename)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Stack dimensions nz, ny, nx (pixels): {:d} x {:d} x {:d}'.format(*stack_shape))
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Stack Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
            layer_ids = np.arange(self.nz_tiles)
            params_mult = []
            for layer_id in layer_ids:
                fls_layer = self.fls[layer_id].ravel()
                tr_matr_layer = self.tr_matr[layer_id]
                params_mult.append([layer_id, fls_layer, image_name, tr_matr_layer, weight_min, weight_max, fill_value, self.shape, self.Xsize, self.Ysize, left_crop, verbose])

            if use_DASK:
                shared_data_future = DASK_client.scatter(deformation_field, broadcast=True)
                futures = DASK_client.map(assemble_layer, params_mult, deformation_field = shared_data_future, retries = DASK_client_retries)
                for future in as_completed(futures):
                    mosaic_out, j = future.result()
                    mrc_new.data[j, :, :] = mosaic_out.astype(dtp)
                    future.cancel()
            else:
                for j, params in enumerate(tqdm(params_mult, desc = 'Saving the data stack into MRC file')):
                    mrc_new.data[j, :, :] = assemble_layer(params, deformation_field)[0].astype(dtp)
            mrc_new.close()
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving Finished')
        return fnms_saved
