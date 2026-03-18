import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
import glob
import re
from scipy.optimize import curve_fit

import matplotlib
import matplotlib.image as mpimg
from matplotlib import pylab, mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

#from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff

from scipy.signal import savgol_filter

from FIBSEM_gs_py.FIBSEM_gs import FIBSEM_frame
from FIBSEM_gs_py.FIBSEM_help_functions_gs import get_min_max_thresholds, find_FWHM, gauss_without_offset


def extract_FFT(fl):
    imga = FIBSEM_frame(fl).RawImageA.astype(float)[0:1250, 0:1250]
    fft_abs = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(imga))))
    return fft_abs

def streak_magnitude(FFT_array, **kwargs):
    '''
    Analyzes FFT array and determines streak magnitude as a peak of averaged cross-section of a selected FFT window. 06/2025. gleb.shtengel@gmail.com
    Assumes that the streaks have very distinct signature in the FFT domain - an increased FFT magnitude along FFT x-axis
    Steps:
    1. Divide the input FFT by transposed input FFT
    2. Select a subset of the above array and average it along X-axis, yielding the streak magnitude array.
    3. Determine the maximum of that array np.max(streak_mag_array)
    
    Parameters:
    ----------
    FFT_array : 2D array
        Input FFT array
    
    kwargs:
    ----------
    FFT_xi : int
        Left boundary of the FFT window for analysis. Default is 1/2 of the FFT_array width.
    FFT_xa : int
        Right boundary of the FFT window for analysis. Default is a 3/4 of the FFT_array width.
    FFT_yi : int
        Top boundary of the FFT window for analysis. Default is 49/100 of the FFT_array width.
    FFT_ya : int
        Bottom boundary of the FFT window for analysis. Default is 51/100 of the FFT_array width.
    disp_res : boolean
        If True, the results will be displayed. Default is False.
    axs : matplotlib axis artists
        If not provided, new plot will be generated.
    titles : titles
        Plot titles. Default is ['Log of Input FFT', 'Analysis window: FFT / FFT $^{T}$'].
    labels : labels
        Plot labels. Default is ['analysis window', 'Averaged FFT / FFT$^{T}$' + ' profile, min={:.2f}, max={:.2f}'.format(np.min(streak_mag_array), np.max(streak_mag_array))].
    locs : strings
        Plot legend locations. Default is ['upper right', 'upper right'].
    v_min_max1 : [float, float]
        List of Min and Max values for FFT / FFT.T image. Default is auto-generated.
    profile_color : str
        Color of the profile plot. Default is 'lime'.
    verbose : boolean
        If True, the intermediate printouts are enabled. Default is False.
    method : str
        Method to find streak magnitude. Options are: 'max' (default), 'mean', and 'fit' (gaussian fit).
    return_all : boolean
        If True, returns streak_magnitude, streak_mag_array, otherwise streak_magnitude. Default is False.
    Returns: streak_magnitude, streak_mag_array
    '''
    
    fsy, fsx = FFT_array.shape
    disp_res = kwargs.get('disp_res', False)
    verbose = kwargs.get('verbose', False)
    method = kwargs.get('method', 'max')
    axs = kwargs.get('axs', '')
    titles = kwargs.get ('titles', ['Log of Input FFT', 'Analysis window: FFT / FFT $^{T}$'])
    locs = kwargs.get('locs', ['upper right', 'upper right'])
    profile_color = kwargs.get('profile_color', 'lime')
    return_all = kwargs.get('return_all', False)
    
    if fsx == fsy:
        FFT_xi = kwargs.get('FFT_xi', fsx//2+fsx//100)
        FFT_xa = kwargs.get('FFT_xa', fsx//4*3)
        FFT_yi = kwargs.get('FFT_yi', fsy//2-fsy//100)
        FFT_ya = kwargs.get('FFT_ya', fsy//2+fsy//100)
        streak_mag = np.abs(np.nan_to_num(FFT_array/FFT_array.T)[FFT_yi:FFT_ya, FFT_xi:FFT_xa])
    else:
        fsm = np.min((fsx, fsy))
        FFT_xi = kwargs.get('FFT_xi', fsm//2+fsm//100)
        FFT_xa = kwargs.get('FFT_xa', np.min((np.max((fsm//4*3, fsx//4*3)), fsy)))
        FFT_yi = kwargs.get('FFT_yi', fsm//2-fsm//100)
        FFT_ya = kwargs.get('FFT_ya', fsm//2+fsm//100)
        FFT_truncated = FFT_array[fsy//2-fsm//2:fsy//2+fsm//2, fsx//2-fsm//2:fsx//2+fsm//2]
        if verbose:
            print('Tiles are not square, analyzing square {:d}x{:d} central subset of FFT'.format(fsm, fsm))
            print(FFT_truncated.shape)
        streak_mag = np.abs(np.nan_to_num((FFT_truncated/FFT_truncated.T)[FFT_yi:FFT_ya, FFT_xi:FFT_xa]))
    streak_mag_array = streak_mag.mean(axis=1)-1.0
    
    if verbose:
        print('Use ' + method + ' to determine streak magnitude')
    streak_magnitude = np.max(streak_mag_array)
    if method == 'mean':
        streak_magnitude = np.mean(streak_mag_array)
    if method == 'fit':
        amp_guess = np.max(streak_mag_array)-np.min(streak_mag_array)
        center_guess = len(streak_mag_array)//2
        sigma_guess = np.min((find_FWHM(np.arange(len(streak_mag_array)), streak_mag_array)[0] / 2.4, 3.0))
        lower_bounds = (amp_guess*0.5, center_guess-2.0, 0.5)
        upper_bounds = (amp_guess*2.0, center_guess+2.0, 5.0)
        try:
            if verbose:
                print('Gaussian Fit Guesses: ', [amp_guess, center_guess, sigma_guess])
                print('Gaussian Fit Bounds: ', (lower_bounds, upper_bounds))
            popt, pcov = curve_fit(gauss_without_offset, np.arange(len(streak_mag_array)), streak_mag_array, p0=[amp_guess, center_guess, sigma_guess], bounds = (lower_bounds, upper_bounds), maxfev=5000)
            streak_magnitude = popt[0]
            if verbose:
                print('Gaussian Fit Coefficients: ', popt)
        except:
            if verbose:
                print('Gaussian Fit did not converge, using max value of {:.2f} instead'.format(streak_magnitude))
            pass
    
    if axs != '' or disp_res:
        if verbose:
            print('Generating output plot')
            
        if axs == '' and disp_res:
            sx = 7.0
            sy = sx * (fsy/fsx + np.abs(FFT_ya-FFT_yi)/np.abs(FFT_xa-FFT_xi))
            fig, axs = plt.subplots(2,1, figsize=(sx, sy), gridspec_kw={'height_ratios':[fsy/fsx, np.abs(FFT_ya-FFT_yi)/np.abs(FFT_xa-FFT_xi)]})
        log_FFT = np.log(np.abs(FFT_array))
        labels = kwargs.get('labels', ['analysis window', 'Mean FFT / FFT$^{T}$' + ' profile, min={:.2f}, max={:.2f}'.format(np.min(streak_mag_array), np.max(streak_mag_array))])
        vmin0, vmax0 = get_min_max_thresholds(log_FFT, disp_res=False)
        axs[0].imshow(log_FFT, cmap='Greys', vmin=vmin0, vmax=vmax0)
        axs[0].axis(False)
        axs[0].set_title(titles[0])
        axs[0].plot([FFT_xi], [FFT_yi], color = 'cyan', linewidth=0.5, label = labels[0])
        if fsx == fsy:
            analysis_window_patch = patches.Rectangle((FFT_xi, FFT_yi), np.abs(FFT_xa-FFT_xi)-2, np.abs(FFT_ya-FFT_yi)-2, linewidth=1.0, edgecolor='cyan',facecolor='none')
        else:
            analysis_window_patch = patches.Rectangle(((fsx//2-fsm//2+FFT_xi), (fsy//2-fsm//2+FFT_yi)), np.abs(FFT_xa-FFT_xi)-2, np.abs(FFT_ya-FFT_yi)-2, linewidth=1.0, edgecolor='cyan',facecolor='none')
            #analysis_window_patch = patches.Rectangle((fsx//2, fsy//2-fsm//40), np.abs(FFT_xa-FFT_xi)-2, np.abs(FFT_ya-FFT_yi)-2, linewidth=1.0, edgecolor='cyan',facecolor='none')
            truncated_window_patch = patches.Rectangle((fsx//2-fsm//2, fsy//2-fsm//2), fsm-2, fsm-2, linewidth=1.0, edgecolor='red',facecolor='none')
            axs[0].add_patch(truncated_window_patch)
            axs[0].plot([fsx//2-fsm//2], [fsy//2-fsm//2], color = 'red', linewidth=0.5, label = 'truncated FFT')
        axs[0].add_patch(analysis_window_patch)
        axs[0].legend(loc=locs[0])
        vmin1, vmax1 = kwargs.get('v_min_max1', get_min_max_thresholds(streak_mag, disp_res=False))
        if verbose:
            print('Using vmin1={:.2f}, vmax1={:.2f}'.format(vmin1, vmax1))
        axs[1].imshow(streak_mag, cmap='inferno', vmin=vmin1, vmax=vmax1)
        axs[1].axis(False)
        axs[1].set_title(titles[1])
        label = labels[1]
        axs[1].plot(streak_mag_array + len(streak_mag_array)//5, np.arange(len(streak_mag_array)), color=profile_color, label = label)
        axs[1].legend(loc=locs[1])
    if return_all:
        return streak_magnitude, streak_mag_array
    else:
        return streak_magnitude


def build_and_analyze_tiled_FFT_map(img, **kwargs):
    '''
    Splits image into tiles and determines FFT and streak magnitude for each tile. 06/2025. gleb.shtengel@gmail.com
    Uses streak_magnitude(FFT_array, **kwargs).
    
    Parameters:
    ----------
    img : 2D array
        Input Image
    
    kwargs:
    ----------
    ntiles_x : int
        Number of FFT tiles in X-direction. Default is 2.
    ntiles_y : int
        Number of FFT tiles in Y-direction. Default is 2.
    determine_streak_magnitudes : boolean
        If True (default), streak magnitude is determined for each tile. Uses streak_magnitude(FFT_array, **kwargs).
    FFT_xi : int
        Left boundary of the FFT window for analysis. Default is 1/2 of the FFT_array width.
    FFT_xa : int
        Right boundary of the FFT window for analysis. Default is a 3/4 of the FFT_array width.
    FFT_yi : int
        Top boundary of the FFT window for analysis. Default is 19/40 of the FFT_array width.
    FFT_ya : int
        Bottom boundary of the FFT window for analysis. Default is 21/40 of the FFT_array width.
    disp_res : boolean
        If True, the results will be displayed. Default is False.
    titles : [str, str]
        Plot titles. Default is ['Image', 'Tiled FFT Map'].
    verbose : boolean
        If True, the intermediate printouts are enabled. Default is False.
    method : str
        Method to find streak magnitude. Options are: 'max' (default), 'mean', and 'fit' (gaussian fit)
    return_all : boolean
        If True, streak_magnitude_map and FFT_map are returned. Otherwise only streak_magnitudes are returned. Default is False.
    
    Returns: streak_magnitude_map, FFT_map
    '''
    ntiles_x = kwargs.get('ntiles_x', 2)
    ntiles_y = kwargs.get('ntiles_y', 2)
    fsy, fsx = img.shape
    fsx_tile = fsx//ntiles_x
    fsy_tile = fsy//ntiles_y
    fs_tile = np.min((fsx_tile, fsy_tile))
    
    FFT_xi = kwargs.get('FFT_xi', fsx_tile//2+fsx_tile//100)
    FFT_xa = kwargs.get('FFT_xa', fsx_tile//4*3)
    FFT_yi = kwargs.get('FFT_yi', fsy_tile//2-fsy_tile//100)
    FFT_ya = kwargs.get('FFT_ya', fsy_tile//2+fsy_tile//100)
    disp_res = kwargs.get('disp_res', False)
    titles = kwargs.get('titles', ['Image', 'Tiled FFT Map'])
    verbose = kwargs.get('verbose', False)
    method = kwargs.get('method', 'max')
    return_all = kwargs.get('return_all', False)
    
    if verbose:
        print('Tiling the image ')
    img_tiled = np.moveaxis(img.reshape((ntiles_y, fsy_tile, ntiles_x, fsx_tile)), 2,1).reshape(ntiles_y * ntiles_x, fsy_tile, fsx_tile)
    if verbose:
        print('Calculating tiled FFTs')
    FFT_tiled = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img_tiled, axes=[1,2]), axes=[1,2]),axes=[1,2])
    if verbose:
        print('Building tiled FFT Map')
    FFT_map = np.moveaxis(FFT_tiled.reshape(ntiles_y, ntiles_x, fsy_tile, fsx_tile), 2, 1).reshape(fsy, fsx)
    if verbose:
        print('Analyzing tiled FFTs to determine streak magnitudes')
        
    if fsx_tile == fsy_tile:
        streak_mag = np.abs(np.nan_to_num(FFT_tiled/FFT_tiled.transpose(0,2,1))[:, FFT_yi:FFT_ya, FFT_xi:FFT_xa])
    else:
        FFT_xi = fs_tile//2+fs_tile//100
        FFT_xa = fs_tile//4*3
        FFT_yi = fs_tile//2-fs_tile//50
        FFT_ya = fs_tile//2+fs_tile//50
        print('Tiles are not square, analyzing square {:d}x{:d} central subsets of FFT for each tile'.format(fs_tile,fs_tile))
        FFT_tiled_truncated = FFT_tiled[:, fsy_tile//2-fs_tile//2:fsy_tile//2+fs_tile//2, fsx_tile//2-fs_tile//2:fsx_tile//2+fs_tile//2]
        print(FFT_tiled_truncated.shape)
        streak_mag = np.abs(np.nan_to_num((FFT_tiled_truncated/FFT_tiled_truncated.transpose(0,2,1))[:, FFT_yi:FFT_ya, FFT_xi:FFT_xa]))
    streak_magnitude_arrays = streak_mag.mean(axis=2)-1.0

    if verbose:
        print('Use ' + method + ' to determine streak magnitude')
    streak_magnitudes = []
    jj=0
    for streak_mag_array in streak_magnitude_arrays:
        streak_magnitude = np.max(streak_mag_array)
        if method == 'mean':
            streak_magnitude = np.mean(streak_mag_array)
        if method == 'fit':
            amp_guess = np.max(streak_mag_array)-np.min(streak_mag_array)
            center_guess = len(streak_mag_array)//2
            sigma_guess = np.min((find_FWHM(np.arange(len(streak_mag_array)), streak_mag_array)[0]/2.4, 4.0))
            lower_bounds = (amp_guess*0.5, center_guess-2.0, 0.5)
            upper_bounds = (amp_guess*2.0, center_guess+2.0, 10.0)
            '''
            if verbose:
                print('Analyzing tile: ', jj)
                print('Use fit parameters: amp_guess={:.2f}, center_guess={:2f}, sigma_guess={:.2f}'.format(amp_guess, center_guess, sigma_guess))
            '''
            try:
                popt, pcov = curve_fit(gauss_without_offset, np.arange(len(streak_mag_array)), streak_mag_array, p0=[amp_guess, center_guess, sigma_guess], bounds=(lower_bounds, upper_bounds), maxfev=5000)
                streak_magnitude = popt[0]
            except:
                if verbose:
                    print('Gaussian Fit did not converge for the tile {:d}, using max value of {:.2f} instead'.format(jj, streak_magnitude))
                pass
        streak_magnitudes.append(streak_magnitude)
        jj+=1
    streak_magnitudes = np.array(streak_magnitudes).reshape(ntiles_y, ntiles_x)

    log_FFT = np.log(np.abs(FFT_map))

    if disp_res:
        if verbose:
            print('Generating output plots')
        fig, axs = plt.subplots(1,2, figsize=(13, 6.8))
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.95,  wspace=0.025)
        vmin0, vmax0 = get_min_max_thresholds(img, disp_res=False)
        axs[0].imshow(img, vmin=vmin0, vmax=vmax0, cmap='Greys')
        xticks = np.arange(0, fsx, fsx_tile)
        yticks = np.arange(0, fsy, fsy_tile)
        vmin1, vmax1 = get_min_max_thresholds(log_FFT[0:fsy//ntiles_y, 0:fsx//ntiles_x], disp_res=False)
        axs[1].imshow(log_FFT, cmap='inferno', vmin=vmin1, vmax=vmax1)
        for title, ax in zip(titles, axs):
            ax.set_title(title)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.grid(which='major', color='yellow')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_frame_on(False)
            ax.tick_params(tick1On=False)
        for j in np.arange(ntiles_y):
            for i in np.arange(ntiles_x):
                axs[0].text((i+0.1)*fsx//ntiles_x, (j+0.2)*fsy//ntiles_y, '{:.2f}'.format(streak_magnitudes[j, i]), color='red')
                axs[1].text((i+0.1)*fsx//ntiles_x, (j+0.2)*fsy//ntiles_y, '{:.2f}'.format(streak_magnitudes[j, i]), color='red')
    if return_all:
        return streak_magnitudes, FFT_map
    else:
        return streak_magnitudes
    
def build_tiled_FFT_map(img, **kwargs):
    '''
    Splits image into tiles, calculates FFT for each tile. 06/2025. gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    img : 2D array
        Input image
    
    kwargs:
    ----------
    ntiles_x : int
        Number of FFT tiles in X-direction. Default is 2.
    ntiles_y : int
        Number of FFT tiles in Y-direction. Default is 2.
    determine_streak_magnitudes : boolean
        If True (default), streak magnitude is determined for each tile. Uses streak_magnitude(FFT_array, **kwargs).
    FFT_xi : int
        Left boundary of the FFT window for analysis. Default is 1/2 of the FFT_array width.
    FFT_xa : int
        Right boundary of the FFT window for analysis. Default is a 3/4 of the FFT_array width.
    FFT_yi : int
        Top boundary of the FFT window for analysis. Default is 19/40 of the FFT_array width.
    FFT_ya : int
        Bottom boundary of the FFT window for analysis. Default is 21/40 of the FFT_array width.
    calc_abs : boolean
        if True, absolute value of FFT is returned. Default is True.
    verbose : boolean
        If True, the intermediate printouts are enabled. Default is False.
    
    Returns: FFT_map
    '''
    ntiles_x = kwargs.get('ntiles_x', 2)
    ntiles_y = kwargs.get('ntiles_y', 2)
    fsy, fsx = img.shape
    fsx_tile = fsx//ntiles_x
    fsy_tile = fsy//ntiles_y
    fs_tile = np.min((fsx_tile, fsy_tile))
    
    FFT_xi = kwargs.get('FFT_xi', fsx_tile//2+fsx_tile//100)
    FFT_xa = kwargs.get('FFT_xa', fsx_tile//4*3)
    FFT_yi = kwargs.get('FFT_yi', fsy_tile//2-fsy_tile//100)
    FFT_ya = kwargs.get('FFT_ya', fsy_tile//2+fsy_tile//100)
    verbose = kwargs.get('verbose', False)
    calc_abs = kwargs.get('calc_abs', True)

    if verbose:
        print('Tiling the image ')
    img_tiled = np.moveaxis(img.reshape((ntiles_y, fsy_tile, ntiles_x, fsx_tile)), 2,1).reshape(ntiles_y * ntiles_x, fsy_tile, fsx_tile)
    if verbose:
        print('Calculating tiled FFTs')
    FFT_tiled = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img_tiled, axes=[1,2]), axes=[1,2]),axes=[1,2])
    if verbose:
        print('Building tiled FFT Map')
    FFT_map = np.moveaxis(FFT_tiled.reshape(ntiles_y, ntiles_x, fsy_tile, fsx_tile), 2, 1).reshape(fsy, fsx)
    if calc_abs:
        FFT_map = np.abs(FFT_map).astype(float)
    return FFT_map


def analyze_tiled_FFT_map(FFT_map, **kwargs):
    '''
    Splits FFT_map into tiles and analyzes the streak magnitude for each tile. 06/2025. gleb.shtengel@gmail.com
    Uses streak_magnitude(FFT_array, **kwargs).
    
    Parameters:
    ----------
    FFT_map : 2D array
        Input FFT map
    
    kwargs:
    ----------
    ntiles_x : int
        Number of FFT tiles in X-direction. Default is 2.
    ntiles_y : int
        Number of FFT tiles in Y-direction. Default is 2.
    FFT_xi : int
        Left boundary of the FFT window for analysis. Default is 1/2 of the FFT_array width.
    FFT_xa : int
        Right boundary of the FFT window for analysis. Default is a 3/4 of the FFT_array width.
    FFT_yi : int
        Top boundary of the FFT window for analysis. Default is 19/40 of the FFT_array width.
    FFT_ya : int
        Bottom boundary of the FFT window for analysis. Default is 21/40 of the FFT_array width.
    disp_res : boolean
        If True, the results will be displayed. Default is False.
    verbose : boolean
        If True, the intermediate printouts are enabled. Default is False.
    method : str
        Method to find streak magnitude. Options are: 'max' (default), 'mean', and 'fit' (gaussian fit)   
    ax : matplotlib axis artist
        If not provided, new plot will be generated
    title : str
        Plot title. Default is 'Tiled FFT Map'.
    Returns: streak_magnitude_map
    '''
    ntiles_x = kwargs.get('ntiles_x', 2)
    ntiles_y = kwargs.get('ntiles_y', 2)
    fsy, fsx = FFT_map.shape
    fsx_tile = fsx//ntiles_x
    fsy_tile = fsy//ntiles_y
    fs_tile = np.min((fsx_tile, fsy_tile))
    
    FFT_xi = kwargs.get('FFT_xi', fsx_tile//2+fsx_tile//100)
    FFT_xa = kwargs.get('FFT_xa', fsx_tile//4*3)
    FFT_yi = kwargs.get('FFT_yi', fsy_tile//2-fsy_tile//100)
    FFT_ya = kwargs.get('FFT_ya', fsy_tile//2+fsy_tile//100)
    disp_res = kwargs.get('disp_res', False)
    ax = kwargs.get('ax', '')
    title = kwargs.get('title', 'Tiled FFT Map')
    verbose = kwargs.get('verbose', False)
    method = kwargs.get('method', 'max')
    return_all = kwargs.get('return_all', False)
    
    if verbose:
        print('Splitting FFT_map into tiles')
    FFT_tiled = np.moveaxis(FFT_map.reshape((ntiles_y, fsy_tile, ntiles_x, fsx_tile)), 2,1).reshape(ntiles_y * ntiles_x, fsy_tile, fsx_tile)
        
    if fsx_tile == fsy_tile:
        streak_mag = np.abs(np.nan_to_num(FFT_tiled/FFT_tiled.transpose(0,2,1))[:, FFT_yi:FFT_ya, FFT_xi:FFT_xa])
    else:
        FFT_xi = fs_tile//2+fs_tile//100
        FFT_xa = fs_tile//4*3
        FFT_yi = fs_tile//2-fs_tile//50
        FFT_ya = fs_tile//2+fs_tile//50
        print('Tiles are not square, analyzing square {:d}x{:d} central subsets of FFT for each tile'.format(fs_tile,fs_tile))
        FFT_tiled_truncated = FFT_tiled[:, fsy_tile//2-fs_tile//2:fsy_tile//2+fs_tile//2, fsx_tile//2-fs_tile//2:fsx_tile//2+fs_tile//2]
        print(FFT_tiled_truncated.shape)
        streak_mag = np.abs(np.nan_to_num((FFT_tiled_truncated/FFT_tiled_truncated.transpose(0,2,1))[:, FFT_yi:FFT_ya, FFT_xi:FFT_xa]))
    streak_magnitude_arrays = streak_mag.mean(axis=2)-1.0

    if verbose:
        print('Use ' + method + ' to determine streak magnitude')
    streak_magnitudes = []
    jj=0
    for streak_mag_array in streak_magnitude_arrays:
        streak_magnitude = np.max(streak_mag_array)
        if method == 'mean':
            streak_magnitude = np.mean(streak_mag_array)
        if method == 'fit':
            amp_guess = np.max(streak_mag_array)-np.min(streak_mag_array)
            center_guess = len(streak_mag_array)//2
            sigma_guess = np.min((find_FWHM(np.arange(len(streak_mag_array)), streak_mag_array)[0]/2.4, 4.0))
            lower_bounds = (amp_guess*0.5, center_guess-2.0, 0.5)
            upper_bounds = (amp_guess*2.0, center_guess+2.0, 10.0)
            '''
            if verbose:
                print('Analyzing tile: ', jj)
                print('Use fit parameters: amp_guess={:.2f}, center_guess={:2f}, sigma_guess={:.2f}'.format(amp_guess, center_guess, sigma_guess))
            '''
            try:
                popt, pcov = curve_fit(gauss_without_offset, np.arange(len(streak_mag_array)), streak_mag_array, p0=[amp_guess, center_guess, sigma_guess], bounds=(lower_bounds, upper_bounds), maxfev=5000)
                streak_magnitude = popt[0]
            except:
                if verbose:
                    print('Gaussian Fit did not converge for the tile {:d}, using max value of {:.2f} instead'.format(jj, streak_magnitude))
                pass
        streak_magnitudes.append(streak_magnitude)
        jj+=1
    streak_magnitudes = np.array(streak_magnitudes).reshape(ntiles_y, ntiles_x)

    log_FFT = np.log(np.abs(FFT_map))

    if ax != '' or disp_res:
        if verbose:
            print('Generating output plot')
            
        if ax == '' and disp_res:
            fig, ax = plt.subplots(1,1, figsize=(7,7))
        xticks = np.arange(0, fsx, fsx_tile)
        yticks = np.arange(0, fsy, fsy_tile)
        vmin1, vmax1 = get_min_max_thresholds(log_FFT[0:fsy//ntiles_y, 0:fsx//ntiles_x], disp_res=False)
        ax.imshow(log_FFT, cmap='inferno', vmin=vmin1, vmax=vmax1)
        ax.set_title(title)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.grid(which='major', color='yellow')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)
        ax.tick_params(tick1On=False)
        for j in np.arange(ntiles_y):
            for i in np.arange(ntiles_x):
                ax.text((i+0.1)*fsx//ntiles_x, (j+0.2)*fsy//ntiles_y, '{:.2f}'.format(streak_magnitudes[j, i]), color='red')
                
    return streak_magnitudes