import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
import glob
import re

import matplotlib
import matplotlib.image as mpimg
from matplotlib import pylab, mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from IPython.core.pylabtools import figsize, getfigs
from PIL import Image as PILImage
from PIL.TiffTags import TAGS

from dask.distributed import Client
from dask.distributed import as_completed
from dask import delayed, compute
import dask.array as da
from dask.diagnostics import ProgressBar

#from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

import skimage
#print(skimage.__version__)
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, warp
try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff
from scipy.signal import savgol_filter
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit as cf

from skimage.feature import blob_dog, blob_log, blob_doh
from sklearn.linear_model import (LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import cv2
import mrcfile
import h5py
import pickle
import webbrowser
from IPython.display import IFrame

EPS = np.finfo(float).eps

try:
    from FIBSEM_gs_py.FIBSEM_help_functions_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_help_functions_gs")

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    warnings.filterwarnings('ignore', category=np.RankWarning)
except:
    warnings.filterwarnings('ignore', category=np.exceptions.RankWarning)



############################################
#
#    helper functions for resolution estimations
#
############################################


############################################
#
#                    BLOB Analysis Algorithm.  Top level function is select_blobs_LoG_analyze_transitions(image, **kwargs):
'''
1.	Create a map of potential blob points for evaluation. Use Laplacian of Gaussian (LoG) as a blob detector. (Difference of Gaussian also works, but LoG seems more robust).

2.	For each point from the set as acquired in Step 1 evaluate transitions along X- and Y- directions:
	•	Extract a square subset around each point.
	•	Extract X- and Y- section profiles of the subset through the center.
	•	Find maximum by parabolic fitting the center portion of the profile. Determine the maximum either from parabolic fit (if fit is valid), or by averaging over a window around the center point. In both cases the BAND parameter determines the aperture.
	•	Find minima to the left and right of the center point. Determine the left and right MIN values by averaging the profile over a window around the minimum point. Again, BAND parameter determines the aperture. (There are 3 BAND parameters - for the left min, center max, and right min searches).
	•	For the Left_MIN, MAX, and Right_MIN determined as above, check for errors. ERROR_FLAG is incremented by values:
		o	0 - no error
		o	1 - x-transition failed min_thr check
		o	2 - y-transition failed min_thr check
		o	4 - x-transition is above transition_high_limit
		o	8 - y-transition is above transition_high_limit
		o	16 - failed to add transition
		o	32 - if selected subset is not a right size (blob too close to edge)
	•	If the transition is "good" (ERROR_FLAG=0), find its 37% and 63% transition points by using nearest neighbor interpolation. Then find 37% to 63% transition distance.
'''
#
############################################


def analyze_blob_transitions(amp, ** kwargs):
    '''
    determins the transition parameters and adds the data to the plot. gleb.shtengel@gmail.com 10/2025.
    
    Parameters:
    amp : cross-section profile
    
    kwargs:
    bounds : list
        List of lists of transition leves for analysis. Deafault is .[0.37, 0.63].
    pixel_size : float
        pixel size. default is 4.0 nm
    bands : list of 3 ints
        list of three ints for the averaging bands for determining the left min, peak, and right min of the cross-section profile.
        Deafault is [5, 3, 5].
    disp_res : boolean
        display the results. Default is False
    ax : artist
        axis artist to plot the histogram to. Default is 0 (no artist).
        In the case ax==0 no plotting is done.
    col : str
        color of cross-setcion lines. Default is 'magenta'.
    label : str
        label of histogram plot. Default is empty string.
    set_yscale : boolean
        Default is False. If True, y_scale setting (below) is enforced.
    y_scale : (float, float)
        a pair of loats for y-axis scaling. Defaiult is (0, 100)
    xpos : float
        offset for horisontal label poition. Default is 1.0.
    cols : list of 4 strings
        colors for . Default is ['brown', 'green', 'red', 'black'].
    pref : str
        prefix string. Default is 'X-'.
    markersize : int
        markersize for symbols. Default is 10.
    fs_labels : int
        fontsize for axis labels. Default is 12.
    verbose : boolean
        print the outputs. Default is False
        
    Returns:
    rise_points, fall_points, [ampi, ampa, amp_max]

    '''

    bounds = kwargs.get('bounds', [0.37, 0.63])
    pixel_size = kwargs.get('pixel_size', 4.0)
    bands = kwargs.get('bands', [5 ,3, 5])
    
    disp_res = kwargs.get('disp_res', False)
    ax = kwargs.get('ax', False)
    col = kwargs.get('col', 'magenta')
    set_yscale = kwargs.get('set_yscale', False)
    y_scale = kwargs.get('y_scale', (0.0, 100.0))
    xpos = kwargs.get('xpos', 1.0)
    cols = kwargs.get('cols', ['brown', 'green', 'red', 'black'])
    pref = kwargs.get('pref', 'X-')
    markersize = kwargs.get('markersize', 10)
    fs_labels = kwargs.get('fs_labels', 12)
    verbose = kwargs.get('verbose', False)
                        
    amp=np.array(amp)
    npts = len(amp)
    xc = npts//2   # define center
     
    dxi, dxc, dxa = bands
    xnm = np.arange(npts) * pixel_size
    
    # find the minimum of the left side of the profile
    indmin_i = np.argmin(amp[0:xc-1])
    indmin_i0 = np.max((0, indmin_i-dxi//2-1))
    xmin_i = xnm[indmin_i0:indmin_i0+dxi]
    amp_i = amp[indmin_i0:indmin_i0+dxi]
    ampi = np.mean(amp_i)
    if verbose:
        print('left min ind: ', indmin_i)
        print('left min X Pts: ', xmin_i)
        print('left min Y Pts: ', amp_i)
        print('left min: ', ampi)
    
    # find the minimum of the right side of the profile
    indmin_a = np.argmin(amp[xc+1:])
    indmin_a1 = np.min((npts, indmin_a+xc+2+dxa//2))
    xmin_a = xnm[indmin_a1-dxa:indmin_a1]
    amp_a = amp[indmin_a1-dxa:indmin_a1]
    ampa = np.mean(amp_a)
    if verbose:
        print('right min ind: ', indmin_a+xc+1)
        print('right min X Pts: ', xmin_a)
        print('right min Y Pts: ', amp_a)
        print('right min: ', ampa)
    
    # find peak by performing parabolic fit around the center
    amp_subset = amp[(xc-dxc//2):(xc+dxc//2+1)]
    xnm_subset = xnm[(xc-dxc//2):(xc+dxc//2+1)]
    pcoeff = np.polyfit(xnm_subset, amp_subset, 2)    
    if pcoeff[0]<0:
        # good fit, use parabolic fit data
        xnm_mx = -0.5 * pcoeff[1] / pcoeff[0]
        # for the MAX, we will take the smaller value between the actual max and parabolic fit max
        amp_max = np.min((np.polyval(pcoeff, xnm_mx), np.max(amp_subset)))
        xnm_subset_fit = np.linspace(np.min(xnm_subset), np.max(xnm_subset), 25)
        amp_subset_fit = np.polyval(pcoeff, xnm_subset_fit)
        if verbose:
            print('X Pts: ', xnm_subset)
            print('Y Pts: ', amp_subset)
            print('Parabolic Fit Coeffs: ', pcoeff)
            print('xnm_mx', xnm_mx)
            print('amp_max: ', amp_max)
    else:
        # bad fit, use averaging around center point instead
        xnm_mx = xnm[xc]
        amp_max = np.mean(amp[(xc-dxc//2):(xc+dxc//2+1)])
        xnm_subset_fit = xnm_subset
        amp_subset_fit = amp_subset *0.0 + amp_max
        if verbose:
            print('X Pts: ', xnm_subset)
            print('Y Pts: ', amp_subset)
            print('Parabolic Fit Failed: ', pcoeff)
            print('xnm_mx', xnm_mx)
            print('amp_max: ', amp_max)
    
    #first, find half point: falling edge
    j=xc-1
    while (j<npts-1) and amp[j]>ampa+(amp_max-ampa)*0.5:
        j += 1
    jhhf = j
   
    yi=ampa+bounds[0]*(amp_max-ampa)
    ya=ampa+bounds[1]*(amp_max-ampa)
    j = jhhf
    while j<npts-1 and amp[j]>yi:
        j += 1
    if (amp[j]-amp[j-1]) != 0:
        xi = xnm[j-1] + (yi-amp[j-1])*(xnm[j]-xnm[j-1])/(amp[j]-amp[j-1])
    else:
        xi = xnm[j-1]
    ja = j
    while j>0 and amp[j]<ya:
        j -= 1
    if (amp[j+1]-amp[j]) != 0:
        xa = xnm[j] + (ya-amp[j])*(xnm[j+1]-xnm[j])/(amp[j+1]-amp[j])
    else:
        xa = xnm[j]
    ji=j
    try: 
        [slope, offs] = np.polyfit(xnm[ji:ja+1], amp[ji:ja+1], 1)
    except:
        slope=0.0
    fall_points = [xi, xa, yi, ya, abs((yi-ya)/slope)]
            
    #second, find half point: rising edge
    j=xc+1
    while (j>0) and amp[j]>ampi+(amp_max-ampi)*0.5:
        j -= 1
    jhhf = j+1
   
    yi=ampi+bounds[0]*(amp_max-ampi)
    ya=ampi+bounds[1]*(amp_max-ampi)
    j = jhhf
    while j>0 and amp[j]>yi:
        j -= 1
    if (amp[j+1]-amp[j]) != 0:
        xi = xnm[j] + (yi-amp[j])*(xnm[j+1]-xnm[j])/(amp[j+1]-amp[j])
    else:
        xi = xnm[j]
    ji = j
    while j<npts-1 and amp[j]<ya:
        j += 1
    if (amp[j]-amp[j-1]) != 0:
        xa = xnm[j-1] + (xnm[j]-xnm[j-1])*(ya-amp[j-1])/(amp[j]-amp[j-1])
    else:
        xa = xnm[j-1]
    ja=j
    try:
        [slope, offs] = np.polyfit(xnm[ji:ja+1], amp[ji:ja+1], 1)
    except:
        slope=0.0
    rise_points = [xi, xa, yi, ya, abs((yi-ya)/slope)]

    if disp_res and ax:
        ax.plot(xnm,amp, color=col, label = pref+'cross-section')
        ax.grid(True)
        ax.set_xlabel('Distance (nm)', fontsize=fs_labels)
        ax.set_ylabel('Amplitude', fontsize=fs_labels)
        ax.tick_params(axis='both', which='major', labelsize=fs_labels)
        if set_yscale:
            ax.set_ylim(y_scale)
        ax.plot(xmin_i, xmin_i*0.0+ampi, color=col, linestyle='--')
        ax.plot(xmin_a, xmin_a*0.0+ampa, color=col, linestyle='--')
        ax.plot(xnm_subset_fit, amp_subset_fit, color=col, linestyle='--')
        ax.plot(xnm_mx, amp_max, 'rd')
        ax.plot([rise_points[0], rise_points[1]], [rise_points[2], rise_points[3]], 'o', color = col, markersize = markersize,
            label = '{:.0f}%-{:.0f}%: {:.2f}nm'.format(bounds[0]*100, bounds[1]*100, abs(xa-xi)))
        ax.plot([fall_points[0], fall_points[1]], [fall_points[2], fall_points[3]], 's', markersize = markersize, color = col,
                label = '{:.0f}%-{:.0f}%: {:.2f}nm'.format(bounds[0]*100, bounds[1]*100, abs(xa-xi)))
    return rise_points, fall_points, [ampi, ampa, amp_max]


def analyze_blob(blob_volume, **kwargs):
    '''
    Evaluate if blob is appropriate to be used for analysis. gleb.shtengel@gmail.com 10/2025.
    
    Parameters:
    -------------
    blob_volume : 3D array
        blob voluime
    
    kwargs:
    -------------
    pixel_size : float
        Pixel size in nm. Default is 4.0. 
    subset_size : int
        Subset size (pixels) for blob / transition analysis. Default is 16.
    bounds : lists
        List of transition limits Default is [0.37, 0.63].
        Example of multiple lists: [[0.33, 0.67], [0.20, 0.80]].
    bands : list of 3 ints
        List of three ints for the averaging bands for determining the left min, peak, and right min of the cross-section profile.
        Default is [5, 3, 5].
    min_thr : float
        Threshold for identifying a 'good' transition (bottom < min_thr* top).
    transition_low_limit : float
        Error flag is incremented by 4 if the determined transition distance is below this value. Default is 0.0.
    transition_high_limit : float
        Error flag is incremented by 8 if the determined transition distance is above this value. Default is 10.0.
        
    Returns:
    tr_result, error_flag

    '''
    pixel_size = kwargs.get('pixel_size', 4.0)
    bounds = kwargs.get('bounds', [0.37, 0.63])
    bands = kwargs.get('bands', [5, 3, 5])
    min_thr = kwargs.get('min_thr', 0.4)
    transition_low_limit = kwargs.get('transition_low_limit', 0.0)
    transition_high_limit = kwargs.get('transition_high_limit', 10.0)
    sz, sy, sx = np.shape(blob_volume)
    error_flag = 0
    tr_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if sz==sy and sz==sx:
        try:
            dx2 = sx//2
            tr_x = analyze_blob_transitions(blob_volume[dx2, dx2, :],
                                    pixel_size = pixel_size,
                                    bounds = bounds,
                                    bands = bands,
                                    disp_res = False)

            tr_y = analyze_blob_transitions(blob_volume[dx2, :, dx2],
                                    pixel_size = pixel_size,
                                    bounds = bounds,
                                    bands = bands,
                                    disp_res=False)

            tr_z = analyze_blob_transitions(blob_volume[:, dx2, dx2],
                                    pixel_size = pixel_size,
                                    bounds = bounds,
                                    bands = bands,
                                    disp_res=False)
            # analyze_blob_transitions returns rise_points, fall_points, [ampi, ampa, amp_max]

            trx1 = abs(tr_x[0][1]- tr_x[0][0])
            trx2 = abs(tr_x[1][1]- tr_x[1][0])
            try1 = abs(tr_y[0][1]- tr_y[0][0])
            try2 = abs(tr_y[1][1]- tr_y[1][0])
            trz1 = abs(tr_z[0][1]- tr_z[0][0])
            trz2 = abs(tr_z[1][1]- tr_z[1][0])
            blob_volume_min = np.min(blob_volume)

            if ((tr_x[2][0]-blob_volume_min) > min_thr*(tr_x[2][2]-blob_volume_min)) or ((tr_x[2][1]-blob_volume_min) >  min_thr*(tr_x[2][2]-blob_volume_min)):
                # ampi > amp_max * min_thr or ampi > amp_max * min_thr for X-transition
                error_flag += 1
            if ((tr_y[2][0]-blob_volume_min) > min_thr*(tr_y[2][2]-blob_volume_min)) or ((tr_y[2][1]-blob_volume_min )> min_thr*(tr_y[2][2]-blob_volume_min)):
                # ampi > amp_max * min_thr or ampi > amp_max * min_thr for Y-transition
                error_flag += 2
            if ((tr_z[2][0]-blob_volume_min) > min_thr*(tr_z[2][2]-blob_volume_min)) or ((tr_z[2][1]-blob_volume_min )> min_thr*(tr_z[2][2]-blob_volume_min)):
                # ampi > amp_max * min_thr or ampi > amp_max * min_thr for Z-transition
                error_flag += 4

            if (trx1 > transition_high_limit) or (trx2 > transition_high_limit) or (abs(tr_x[0][4]) > transition_high_limit) or (abs(tr_x[1][4]) > transition_high_limit):
                error_flag += 8
            if (try1 > transition_high_limit) or (try2 > transition_high_limit) or (abs(tr_y[0][4]) > transition_high_limit) or (abs(tr_y[1][4]) > transition_high_limit):
                error_flag += 16
            if (trz1 > transition_high_limit) or (trz2 > transition_high_limit) or (abs(tr_z[0][4]) > transition_high_limit) or (abs(tr_z[1][4]) > transition_high_limit):
                error_flag += 32
            try:              
                tr_result = [(tr_x[2][2]+tr_y[2][2] + tr_z[2][2])/3.0, trx1, trx2, try1, try2, trz1, trz2, abs(tr_x[0][4]), abs(tr_x[1][4]), abs(tr_y[0][4]), abs(tr_y[1][4]), abs(tr_z[0][4]), abs(tr_z[1][4])]
            except:
                error_flag += 64
        except:
            error_flag += 128
    else:
        error_flag += 256
    return tr_result, error_flag


def select_blobs_LoG_analyze_transitions(image, **kwargs):
    '''
    Finds blobs in the given grayscale image using Laplasian of Gaussians (LoG). gleb.shtengel@gmail.com 06/2023
    
    Parameters:
    ---------
    image : 2D image
    
    kwargs:
    ---------
    min_sigma : float
        Min sigma (in pixel units) for Gaussian kernel in LoG search. Default is 1.0.
    max_sigma : float
        Max sigma (in pixel units) for Gaussian kernel in LoG search. Default is 1.5.
    threshold : float
        Threshold for LoG search. Default is 0.005. The absolute lower bound for scale space maxima.
        Local maxima smaller than threshold are ignored. Reduce this to detect blobs with less intensities. 
    overlap : float
        A value between 0 and 1. Defaults is 0.1. If the area of two blobs overlaps by a
        fraction greater than 'overlap', the smaller blob is eliminated.
    pixel_size : float
        Pixel size in nm. Default is 4.0. 
    subset_size : int
        Subset size (pixels) for blob / transition analysis. Default is 16.
    bounds : lists
        List of transition limits Default is [0.37, 0.63].
        Example of multiple lists: [[0.33, 0.67], [0.20, 0.80]].
    bands : list of 3 ints
        List of three ints for the averaging bands for determining the left min, peak, and right min of the cross-section profile.
        Default is [5 ,3, 5].
    min_thr : float
        Threshold for identifying a 'good' transition (bottom < min_thr* top).
    transition_low_limit : float
        Error flag is incremented by 4 if the determined transition distance is below this value. Default is 0.0.
    transition_high_limit : float
        Error flag is incremented by 8 if the determined transition distance is above this value. Default is 10.0.
    verbose : boolean
        Print intermediate outputs. Default is False.
    disp_res : boolean
        Display results. Default is False.
    disp_all_blobs : boolean
        Display only good blobs. If True (Default) - all blobs are displayed. Usufull when there are too many bad blobs.
    title : str
        Title string.
    nbins : int
        Number of bins for histogram. Default is 64.
    save_data_xlsx : boolean
        save the data into Excel workbook. Default is True.
    save_good_blobs_only : boolean
        save only good blob data (useful if the dataset is too big). Default is False.
    results_file_xlsx : file name for Excel workbook to save the results
        
    Returns: results_file_xlsx, blobs_LoG, error_flags, tr_results, hst_datas
        results_file_xlsx : name of the Excel workbook with the results
        blobs_LoG : list of lists - all blobs from Step 1
            [y, x, r] for each blob
        error_flags : list of error flags
            error_flag : int
            0 - no error
            1 - x-transition failed min_thr check
            2 - y-transition failed min_thr check
            4 - x-transition is above transition_high_limit
            8 - y-transition is above transition_high_limit
            16 - if failed to add transition
            32 - if selected subset is not a right size (blob too close to edge)
        tr_results : list of lists
            [tr_xs_pts, tr_ys_pts, tr_xs_slp, tr_ys_slp]
        hst_datas : list of arrays of stats results
            rows: X- and Y- transitions, X-transitions, Y-transitions
            columns: Hist.Peak, Median, Mean, STD.
            For example, X-Y hist median is hst_data[0,1]
    '''
    min_sigma = kwargs.get('min_sigma', 1.0)
    max_sigma = kwargs.get('max_sigma', 1.0)
    threshold = kwargs.get('threshold', 0.005)
    overlap = kwargs.get('overlap', 0.1)
    subset_size = kwargs.get('subset_size', 16)     # blob analysis window size in pixels
    dx2=subset_size//2
    pixel_size = kwargs.get('pixel_size', 4.0)
    bounds = kwargs.get('bounds', [0.37, 0.63])
    bands = kwargs.get('bands', [5, 3, 5])        # bands for finding left minimum, mid (peak), and right minimum
    min_thr = kwargs.get('min_thr', 0.4)        #threshold for identifying 'good' transition (bottom < min_thr* top)
    transition_low_limit = kwargs.get('transition_low_limit', 0.0)
    transition_high_limit = kwargs.get('transition_high_limit', 10.0)
    verbose = kwargs.get('verbose', True)
    disp_res = kwargs.get('disp_res', True)
    disp_all_blobs = kwargs.get('disp_all_blobs', True)
    title = kwargs.get('title', '')
    nbins = kwargs.get('nbins', 64)
    save_data_xlsx = kwargs.get('save_data_xlsx', True)
    save_good_blobs_only = kwargs.get('save_good_blobs_only', False)
    results_file_xlsx = kwargs.get('results_file_xlsx', 'results.xlsx')
    if not save_data_xlsx:
        results_file_xlsx = 'Data not saved'
    kwargs['min_sigma'] = min_sigma
    kwargs['max_sigma'] = max_sigma
    kwargs['threshold'] = threshold
    kwargs['overlap'] = overlap
    kwargs['subset_size'] = subset_size
    kwargs['pixel_size'] = pixel_size
    kwargs['bounds'] = bounds
    kwargs['bands'] = bands
    kwargs['min_thr'] = min_thr
    kwargs['transition_low_limit'] = transition_low_limit
    kwargs['transition_high_limit'] = transition_high_limit
    kwargs['verbose'] = verbose
    kwargs['title'] = title
    kwargs['nbins'] = nbins
    kwargs['results_file_xlsx'] = results_file_xlsx
    if verbose:
        print('Step1: Searching for Blobs using Laplasian of Gaussians using kwargs:')
        print(kwargs)
    blobs_LoG = blob_log(image, min_sigma = min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap)
    if verbose:
        print('Step1: Search Blobs using Laplasian of Gaussians, found {:d} blobs'.format(len(blobs_LoG)))

    error_flags = []
    tr_results = []
    subset_mags = []
    hst_datas = []

    for j, blob in enumerate(tqdm(blobs_LoG, desc='Sortings blobs by magnitude', display=verbose)):
        y, x, r = blob
        xc = int(x)
        yc = int(y)
        subset_mags.append(np.mean(image[yc-1:yc+1, xc-1:xc+1]))
    
    subset_mags = np.array(subset_mags)
    ind_sorted = np.flip(np.argsort(subset_mags))
    blobs_LoG = np.array(blobs_LoG)[ind_sorted]
    
    for j, blob in enumerate(tqdm(blobs_LoG, desc='Analyzing blobs', display=verbose)):
        error_flag = 0
        tr_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            y, x, r = blob
            xc = int(x)
            yc = int(y)
            subset = image[yc-dx2:yc+dx2, xc-dx2:xc+dx2]
            if np.shape(subset)==(subset_size, subset_size):
                tr_x = analyze_blob_transitions(subset[dx2, :],
                                        pixel_size = pixel_size,
                                        bounds = bounds,
                                        bands = bands,
                                        disp_res = False)

                tr_y = analyze_blob_transitions(subset[:, dx2],
                                        pixel_size = pixel_size,
                                        bounds = bounds,
                                        bands = bands,
                                        disp_res=False)
                # analyze_blob_transitions returns rise_points, fall_points, [ampi, ampa, amp_max]

                trx1 = abs(tr_x[0][1]- tr_x[0][0])
                trx2 = abs(tr_x[1][1]- tr_x[1][0])
                try1 = abs(tr_y[0][1]- tr_y[0][0])
                try2 = abs(tr_y[1][1]- tr_y[1][0])
                subset_min = np.min(subset)

                if ((tr_x[2][0]-subset_min) > min_thr*(tr_x[2][2]-subset_min)) or ((tr_x[2][1]-subset_min) >  min_thr*(tr_x[2][2]-subset_min)):
                    # ampi > amp_max * min_thr or ampi > amp_max * min_thr for X-transition
                    error_flag += 1
                if ((tr_y[2][0]-subset_min) > min_thr*(tr_y[2][2]-subset_min)) or ((tr_y[2][1]-subset_min )> min_thr*(tr_y[2][2]-subset_min)):
                    # ampi > amp_max * min_thr or ampi > amp_max * min_thr for Y-transition
                    error_flag += 2
                if (trx1 > transition_high_limit) or (trx2 > transition_high_limit) or (abs(tr_x[0][4]) > transition_high_limit) or (abs(tr_x[1][4]) > transition_high_limit):
                    error_flag += 4
                if (try1 > transition_high_limit) or (try2 > transition_high_limit) or (abs(tr_y[0][4]) > transition_high_limit) or (abs(tr_y[1][4]) > transition_high_limit):
                    error_flag += 8
                try:              
                    tr_result = [(tr_x[2][2]+tr_y[2][2])/2.0, trx1, trx2, try1, try2, abs(tr_x[0][4]), abs(tr_x[1][4]), abs(tr_y[0][4]), abs(tr_y[1][4])]
                except:
                    error_flag += 16
                    #if verbose:
                    #    print('could not append blob, error_flag={:d}'.format(error_flag))
            else:
                error_flag += 32
                #if verbose:
                #    print('could not append blob, error_flag={:d}'.format(error_flag))
        except:
            error_flag += 64
        tr_results.append(tr_result)
        error_flags.append(error_flag)
        
    error_flags = np.array(error_flags)
    tr_results = np.array(tr_results)
    if len(error_flags[error_flags==0]) > 0:
    #blobs_LoG = blobs_LoG[error_flags==0]
        Xpt1 = tr_results[error_flags==0, 1]
        Xpt2 = tr_results[error_flags==0, 2]
        Ypt1 = tr_results[error_flags==0, 3]
        Ypt2 = tr_results[error_flags==0, 4]
        Xslp1 = tr_results[error_flags==0, 5]
        Xslp2 = tr_results[error_flags==0, 6]
        Yslp1 = tr_results[error_flags==0, 7]
        Yslp2 = tr_results[error_flags==0, 8]
        XYpt_selected = [Xpt1, Xpt2, Ypt1, Ypt2]
        Xpt_selected = [Xpt1, Xpt2]
        Ypt_selected = [Ypt1, Ypt2]
        Xslp_selected = [Xslp1, Xslp2]
        Yslp_selected = [Yslp1, Yslp2]
        tr_mean = np.mean(XYpt_selected)
        tr_std = np.std(XYpt_selected)
        tr_sets = [[Xpt_selected, Ypt_selected],
            [Xslp_selected, Yslp_selected]]

        if save_data_xlsx:
            xlsx_writer = pd.ExcelWriter(results_file_xlsx, engine='xlsxwriter')
            trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
            columns=['Y', 'X', 'R', 'Amp',
                trans_str + ' X-pt1',
                trans_str + ' X-pt2',
                trans_str + ' Y-pt1',
                trans_str + ' Y-pt2',
                trans_str + ' X-slp1',
                trans_str + ' X-slp2',
                trans_str + ' Y-slp1',
                trans_str + ' Y-slp2',
                'error_flag']
            blobs_LoG_arr = np.array(blobs_LoG)
            if save_good_blobs_only:
                transition_results = pd.DataFrame(np.column_stack((blobs_LoG_arr[error_flags==0], tr_results[error_flags==0, :], error_flags[error_flags==0])), columns = columns, index = None)
            else:
                transition_results = pd.DataFrame(np.column_stack((blobs_LoG_arr, tr_results, error_flags)), columns = columns, index = None)                
            transition_results.to_excel(xlsx_writer, index=None, sheet_name='Transition analysis results')
            kwargs_info = pd.DataFrame([kwargs]).T # prepare to be save in transposed format
            kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')
        
        fexts =['_{:.0f}{:.0f}pts'.format(bounds[0]*100, bounds[1]*100), '_{:.0f}{:.0f}slp'.format(bounds[0]*100, bounds[1]*100)]
        sheet_names = ['{:.0f}%-{:.0f}% summary (pts)'.format(bounds[0]*100, bounds[1]*100),
            '{:.0f}%-{:.0f}% summary (slopes)'.format(bounds[0]*100, bounds[1]*100)]    

        hranges = [(0, 10.0), 
               (0, 10.0)]  # histogram range for the transition distance (in nm))
        for [tr_xs, tr_ys], fext, sheet_name, hrange in zip(tr_sets, fexts, sheet_names, hranges):
            trs = np.squeeze(np.array((tr_xs, tr_ys)).flatten())
            tr_x = np.array(tr_xs).flatten()
            tr_y = np.array(tr_ys).flatten()
            dsets = [trs, tr_x, tr_y]
            hst_data =  np.zeros((len(dsets), 4))
            cc_data =  np.zeros((nbins, len(dsets)+1))
            for (j, dset) in enumerate(dsets):
                hst_res = np.array(add_hist(dset, nbins=nbins, hrange=hrange, ax=0, col='red', label=''), dtype=object)
                #add_hist returns x, y, histmax_x, md, mn, std
                hst_data[j, :] = hst_res[2:6]
                cc_data[:, j+1] = hst_res[1]
            hst_datas.append(hst_data)
            cell_text = [['{:.2f}'.format(d) for d in dd] for dd in hst_data]
            if save_data_xlsx:
                columns = ['Hist. Peak', 'Median', 'Mean', 'STD']
                rows = ['X, Y', 'X', 'Y']
                n_cols = len(columns)
                n_rows = len(rows)
                transition_summary = pd.DataFrame(hst_data, columns = columns, index = None)
                transition_summary.insert(0, '', rows)
                transition_summary.to_excel(xlsx_writer, index=None, sheet_name=sheet_name)
        if save_data_xlsx:
            #xlsx_writer.save()
            xlsx_writer.close()
    else:
        Xpt1 = []
        Xpt2 = []
        Ypt1 = []
        Ypt2 = []
        Xslp1 = []
        Xslp2 = []
        Yslp1 = []
        Yslp2 = []
        XYpt_selected = []
        Xpt_selected = []
        Ypt_selected = []
        Xslp_selected = []
        Yslp_selected = []
        tr_mean = 0.0
        tr_std = 0.0
        tr_sets = [[Xpt_selected, Ypt_selected],
            [Xslp_selected, Yslp_selected]]
        save_data_xlsx = False
        results_file_xlsx = 'Data not saved'

    if verbose and len(error_flags[error_flags==0]) > 0:
        print('Step2: Analyzed Blobs/Transitions, selected {:d}'.format(len(blobs_LoG)))
        print('Step2: Analyzed selected {:d} Blobs, found {:d} good ones'.format(len(error_flags), len(error_flags[error_flags==0])))
        if save_data_xlsx:
            print('Step3: Saving the results into file:  ' + results_file_xlsx)
        else:
            print('Data is NOT saved')
        
    if verbose and disp_res:
        print('Step4: Displaying the blob map')
    if disp_res:
        szy, szx = image.shape
        sx = 15.0
        sy = sx / szx * szy
        fig, ax = plt.subplots(1, 1, figsize=(sx, sy))
        vmin, vmax = get_min_max_thresholds(image, disp_res=False)
        ax.imshow(image, cmap='Greys', vmin=vmin, vmax=vmax)
        ax.axis(False)
        for blob, error_flag in zip(blobs_LoG, error_flags):
            y, x, r = blob
            if error_flag == 0:
                colr = 'lime'
                c = plt.Circle((x, y), r*3, color=colr, linewidth=0.5, fill=False)
                ax.add_patch(c)
            else:
                colr = 'blue'
                if disp_all_blobs:
                    c = plt.Circle((x, y), r*3, color=colr, linewidth=0.5, fill=False)
                    ax.add_patch(c)
        ax.set_title(title)
        if disp_all_blobs:
            ax.text(0.02,0.97, 'o  All Blobs', color = 'blue', fontsize = 12, transform=ax.transAxes)
        ax.text(0.02,0.95, 'o  Good Blobs', color = 'lime', fontsize = 12, transform=ax.transAxes)
    return results_file_xlsx, blobs_LoG, error_flags, tr_results, hst_datas


def select_blobs_LoG_analyze_transitions_3D(volume, **kwargs):
    '''
    Finds blobs in the given grayscale image using Laplasian of Gaussians (LoG). gleb.shtengel@gmail.com 10/2025
    Uses DASK delayed, not DASK Client, so no need to provide the DASK client info.
    
    Parameters:
    ---------
    volume : 3D array
    
    kwargs:
    ---------
    min_sigma : float
        Min sigma (in pixel units) for Gaussian kernel in LoG search. Default is 1.0.
    max_sigma : float
        Max sigma (in pixel units) for Gaussian kernel in LoG search. Default is 1.5.
    threshold : float
        Threshold for LoG search. Default is 0.005. The absolute lower bound for scale space maxima.
        Local maxima smaller than threshold are ignored. Reduce this to detect blobs with less intensities. 
    overlap : float
        A value between 0 and 1. Defaults is 0.1. If the area of two blobs overlaps by a
        fraction greater than 'overlap', the smaller blob is eliminated.
    pixel_size : float
        Pixel size in nm. Default is 4.0. 
    subset_size : int
        Subset size (pixels) for blob / transition analysis. Default is 16.
    bounds : lists
        List of transition limits Default is [0.37, 0.63].
        Example of multiple lists: [[0.33, 0.67], [0.20, 0.80]].
    bands : list of 3 ints
        List of three ints for the averaging bands for determining the left min, peak, and right min of the cross-section profile.
        Default is [5, 3, 5].
    min_thr : float
        Threshold for identifying a 'good' transition (bottom < min_thr* top).
    transition_low_limit : float
        Error flag is incremented by 4 if the determined transition distance is below this value. Default is 0.0.
    transition_high_limit : float
        Error flag is incremented by 8 if the determined transition distance is above this value. Default is 10.0.
    chunk_size : int
        chunk size for splitting the image. Default is 64.
    border_depth : int
        Size of chunk overlap. Default is int(4*max_sigma)
    verbose : boolean
        Print intermediate outputs. Default is False.
    nbins : int
        Number of bins for histogram. Default is 64.
    save_data_xlsx : boolean
        save the data into Excel workbook. Default is True.
    save_good_blobs_only : boolean
        save only good blob data (useful if the dataset is too big). Default is False.
    results_file_xlsx : file name for Excel workbook to save the results
        
    Returns: results_file_xlsx, blobs_LoG, error_flags, tr_results, hst_datas
        results_file_xlsx : name of the Excel workbook with the results
        blobs_LoG : list of lists - all blobs from Step 1
            [y, x, r] for each blob
        error_flags : list of error flags
            error_flag : int
            0 - no error
            1 - x-transition failed min_thr check
            2 - y-transition failed min_thr check
            4 - x-transition is above transition_high_limit
            8 - y-transition is above transition_high_limit
            16 - if failed to add transition
            32 - if selected subset is not a right size (blob too close to edge)
        tr_results : list of lists
            [tr_xs_pts, tr_ys_pts, tr_xs_slp, tr_ys_slp]
        hst_datas : list of arrays of stats results
            rows: X- and Y- transitions, X-transitions, Y-transitions
            columns: Hist.Peak, Median, Mean, STD.
            For example, X-Y hist median is hst_data[0,1]
    '''
    min_sigma = kwargs.get('min_sigma', 1.0)
    max_sigma = kwargs.get('max_sigma', 1.5)
    threshold = kwargs.get('threshold', 0.005)
    overlap = kwargs.get('overlap', 0.1)
    subset_size = kwargs.get('subset_size', 16)     # blob analysis window size in pixels
    dx2=subset_size//2
    pixel_size = kwargs.get('pixel_size', 4.0)
    bounds = kwargs.get('bounds', [0.37, 0.63])
    bands = kwargs.get('bands', [5, 3, 5])        # bands for finding left minimum, mid (peak), and right minimum
    min_thr = kwargs.get('min_thr', 0.4)        #threshold for identifying 'good' transition (bottom < min_thr* top)
    transition_low_limit = kwargs.get('transition_low_limit', 0.0)
    transition_high_limit = kwargs.get('transition_high_limit', 10.0)
    verbose = kwargs.get('verbose', False)
    nbins = kwargs.get('nbins', 64)
    save_data_xlsx = kwargs.get('save_data_xlsx', True)
    save_good_blobs_only = kwargs.get('save_good_blobs_only', False)
    results_file_xlsx = kwargs.get('results_file_xlsx', 'results.xlsx')
    if not save_data_xlsx:
        results_file_xlsx = 'Data not saved'
    border_depth = kwargs.get('border_depth', int(4*max_sigma))
    chunk_size = kwargs.get('chunk_size', 64)
    depth = {0: border_depth, 1: border_depth, 2: border_depth}
    
    kwargs['min_sigma'] = min_sigma
    kwargs['max_sigma'] = max_sigma
    kwargs['threshold'] = threshold
    kwargs['overlap'] = overlap
    kwargs['subset_size'] = subset_size
    kwargs['pixel_size'] = pixel_size
    kwargs['bounds'] = bounds
    kwargs['bands'] = bands
    kwargs['min_thr'] = min_thr
    kwargs['transition_low_limit'] = transition_low_limit
    kwargs['transition_high_limit'] = transition_high_limit
    kwargs['border_depth'] = border_depth
    kwargs['chunk_size'] = chunk_size
    kwargs['verbose'] = verbose
    kwargs['nbins'] = nbins
    kwargs['results_file_xlsx'] = results_file_xlsx
    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Step1: Searching for Blobs using Laplasian of Gaussians with following kwargs:')
        print(kwargs)
        print('Using DASK delayed')
    
    volume_dask0 = da.from_array(volume, chunks=chunk_size)
    volume_dask = da.overlap.overlap(volume_dask0, depth=depth,
                      boundary={0: 'nearest', 1: 'nearest', 2: 'nearest'})
    @delayed
    def process_chunk(chunk, chunk_id, depth, min_sigma, max_sigma, threshold):
        z_offset = (chunk_id[0]) * (chunk.shape[0] - 2*depth[0]) - depth[0]
        y_offset = (chunk_id[1]) * (chunk.shape[1] - 2*depth[1]) - depth[1]
        x_offset = (chunk_id[2]) * (chunk.shape[2] - 2*depth[2]) - depth[2]
        chunk_np = np.array(chunk)
        blobs = blob_log(chunk_np, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, overlap=overlap, exclude_border=(depth[0], depth[1], depth[2]))
        # Adjust coordinates based on chunk position
        if blobs.size > 0:
            blobs[:, 0] += z_offset
            blobs[:, 1] += y_offset
            blobs[:, 2] += x_offset
        return blobs

    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Setting up DASK computations: blob detection')
    blob_futures = []
    for z, chunk_z in enumerate(volume_dask.to_delayed()):
        for y, chunk_y in enumerate(chunk_z):
            for x, chunk in enumerate(chunk_y):
                blob_futures.append(process_chunk(chunk, (z, y, x), depth, min_sigma, max_sigma, threshold))
    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Starting DASK computations: blob detection')
    with ProgressBar():
        results = compute(*blob_futures)
    # Concatenate results
    blobs_LoG = np.vstack([r for r in results if r.size > 0])
    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Step1: Search Blobs using Laplasian of Gaussians, found {:d} blobs'.format(len(blobs_LoG)))

    error_flags = []
    tr_results = []
    subset_mags = []
    hst_datas = []

    for j, blob in enumerate(tqdm(blobs_LoG, desc='Step2: Sortings blobs by magnitude', display=verbose)):
        z, y, x, r = blob
        xc = int(x)
        yc = int(y)
        zc = int(z)
        subset_mags.append(np.mean(volume[zc-1:zc+1, yc-1:yc+1, xc-1:xc+1]))
    
    subset_mags = np.array(subset_mags)
    ind_sorted = np.flip(np.argsort(subset_mags))
    blobs_LoG = np.array(blobs_LoG)[ind_sorted]
    
    for j, blob in enumerate(tqdm(blobs_LoG, desc='Step2: Analyzing blobs', display=verbose)):
        z, y, x, r = blob
        xc = int(x)
        yc = int(y)
        zc = int(z)
        subset = volume[zc-dx2:zc+dx2, yc-dx2:yc+dx2, xc-dx2:xc+dx2]
        tr_result, error_flag = analyze_blob(subset,
                                             pixel_size = pixel_size,
                                             bounds = bounds,
                                             bands = bands,
                                             min_thr = min_thr,
                                             transition_low_limit = transition_low_limit,
                                             transition_high_limit = transition_high_limit)
        tr_results.append(tr_result)
        error_flags.append(error_flag)
        
    error_flags = np.array(error_flags)
    tr_results = np.array(tr_results)
    if len(error_flags[error_flags==0]) > 0:
    #blobs_LoG = blobs_LoG[error_flags==0]
        Xpt1 = tr_results[error_flags==0, 1]
        Xpt2 = tr_results[error_flags==0, 2]
        Ypt1 = tr_results[error_flags==0, 3]
        Ypt2 = tr_results[error_flags==0, 4]
        Zpt1 = tr_results[error_flags==0, 5]
        Zpt2 = tr_results[error_flags==0, 6]
        Xslp1 = tr_results[error_flags==0, 7]
        Xslp2 = tr_results[error_flags==0, 8]
        Yslp1 = tr_results[error_flags==0, 9]
        Yslp2 = tr_results[error_flags==0, 10]
        Zslp1 = tr_results[error_flags==0, 11]
        Zslp2 = tr_results[error_flags==0, 12]
        XYpt_selected = [Xpt1, Xpt2, Ypt1, Ypt2]
        Xpt_selected = [Xpt1, Xpt2]
        Ypt_selected = [Ypt1, Ypt2]
        Zpt_selected = [Zpt1, Zpt2]
        Xslp_selected = [Xslp1, Xslp2]
        Yslp_selected = [Yslp1, Yslp2]
        Zslp_selected = [Zslp1, Zslp2]
        tr_mean = np.mean(XYpt_selected)
        tr_std = np.std(XYpt_selected)
        tr_sets = [[Xpt_selected, Ypt_selected, Zpt_selected],
            [Xslp_selected, Yslp_selected, Zslp_selected]]

        if save_data_xlsx:
            xlsx_writer = pd.ExcelWriter(results_file_xlsx, engine='xlsxwriter')
            trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
            columns=['Z', 'Y', 'X', 'R', 'Amp',
                trans_str + ' X-pt1',
                trans_str + ' X-pt2',
                trans_str + ' Y-pt1',
                trans_str + ' Y-pt2',
                trans_str + ' Z-pt1',
                trans_str + ' Z-pt2',
                trans_str + ' X-slp1',
                trans_str + ' X-slp2',
                trans_str + ' Y-slp1',
                trans_str + ' Y-slp2',
                trans_str + ' Z-slp1',
                trans_str + ' Z-slp2',
                'error_flag']
            if save_good_blobs_only:
                if verbose:
                    print('Saving only good blob data')
                transition_results = pd.DataFrame(np.column_stack((np.array(blobs_LoG[error_flags==0]), tr_results[error_flags==0, :], error_flags[error_flags==0])), columns = columns, index = None)
            else:
                if verbose:
                    print('Saving all blob data')
                transition_results = pd.DataFrame(np.column_stack((np.array(blobs_LoG), tr_results, error_flags)), columns = columns, index = None)
            transition_results.to_excel(xlsx_writer, index=None, sheet_name='Transition analysis results')
            kwargs_info = pd.DataFrame([kwargs]).T # prepare to be save in transposed format
            kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')
        
        fexts =['_{:.0f}{:.0f}pts'.format(bounds[0]*100, bounds[1]*100), '_{:.0f}{:.0f}slp'.format(bounds[0]*100, bounds[1]*100)]
        sheet_names = ['{:.0f}%-{:.0f}% summary (pts)'.format(bounds[0]*100, bounds[1]*100),
            '{:.0f}%-{:.0f}% summary (slopes)'.format(bounds[0]*100, bounds[1]*100)]    

        hranges = [(0, 10.0), 
               (0, 10.0)]  # histogram range for the transition distance (in nm))
        for [tr_xs, tr_ys, tr_zs], fext, sheet_name, hrange in zip(tr_sets, fexts, sheet_names, hranges):
            tr_x = np.array(tr_xs).flatten()
            tr_y = np.array(tr_ys).flatten()
            tr_z = np.array(tr_zs).flatten()
            dsets = [tr_x, tr_y, tr_z]
            hst_data =  np.zeros((len(dsets), 4))
            cc_data =  np.zeros((nbins, len(dsets)+1))
            for (j, dset) in enumerate(dsets):
                hst_res = np.array(add_hist(dset, nbins=nbins, hrange=hrange, ax=0, col='red', label=''), dtype=object)
                hst_data[j, :] = hst_res[2:6]
                cc_data[:, j+1] = hst_res[1]
            hst_datas.append(hst_data)
            cell_text = [['{:.2f}'.format(d) for d in dd] for dd in hst_data]
            if save_data_xlsx:
                columns = ['Hist. Peak', 'Median', 'Mean', 'STD']
                rows = ['X', 'Y', 'Z']
                n_cols = len(columns)
                n_rows = len(rows)
                transition_summary = pd.DataFrame(hst_data, columns = columns, index = None)
                transition_summary.insert(0, '', rows)
                transition_summary.to_excel(xlsx_writer, index=None, sheet_name=sheet_name)
        if save_data_xlsx:
            xlsx_writer.close()
    else:
        Xpt1 = []
        Xpt2 = []
        Ypt1 = []
        Ypt2 = []
        Zpt1 = []
        Zpt2 = []
        Xslp1 = []
        Xslp2 = []
        Yslp1 = []
        Yslp2 = []
        Zslp1 = []
        Zslp2 = []
        XYpt_selected = []
        Xpt_selected = []
        Ypt_selected = []
        Zpt_selected = []
        Xslp_selected = []
        Yslp_selected = []
        Zslp_selected = []
        tr_mean = 0.0
        tr_std = 0.0
        tr_sets = [[Xpt_selected, Ypt_selected, Zpt_selected],
            [Xslp_selected, Yslp_selected, Zslp_selected]]
        save_data_xlsx = False
        results_file_xlsx = 'Data not saved'

    if verbose and len(error_flags[error_flags==0]) > 0:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Step2: Analyzed Blobs/Transitions, selected {:d}'.format(len(blobs_LoG)))
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Step2: Analyzed selected {:d} Blobs, found {:d} good ones'.format(len(error_flags), len(error_flags[error_flags==0])))
        if save_data_xlsx:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Step3: Saving the results into file:  ' + results_file_xlsx)
        else:
            print('Data is NOT saved')
    else:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+' Step2: Analyzed selected {:d} Blobs, found {:d} good ones'.format(len(error_flags), len(error_flags[error_flags==0])))
        print('Data is NOT saved')

    if save_good_blobs_only:
        return results_file_xlsx, blobs_LoG[error_flags==0], error_flags[error_flags==0], tr_results[error_flags==0], hst_datas
    else:
        return results_file_xlsx, blobs_LoG, error_flags, tr_results, hst_datas



############################################
#
#                    EDGE Analysis Algorithm.  Top level function is analyze_edge_transitions_image(image, **kwargs)::
'''
1. Create a map of potential transition points for evaluation. Use one of two paths:
    - Calculate gradient map (and a map of absolute value of a gradient). Order points in in order of absolute values of the gradient.
    
    OR
    
    - Use Canny Edge detector to create a map of potential points. Also Calculate the gradient map. Then order the Canny edge points in order of absolute values of the gradient.
2. Form the map of transition points determined in the Step 1 build a set of points using following procedure:
    - Select a point with maximum absolute value of the gradient and add to set
    - Draw exclusion circle around it (value set by NEIGHBOUR_EXCLUSION_RADIUS), and remove all points withing that circle from further consideration.
    - Go to the next point and repeat.
3. For the points (center_points) in the set as acquired in Step 2 evaluate transitions:
    - transition direction is determined by the local gradient (averaged over a vicinity of points near the selected center_point)
    - build a square subset of the image around the center_point (the size is set by SUBSET_SIZE). Find MIN_CRITERION and MAX_CRITERION - the minimum and maximun image values in the subset (by default 0.2 of CDF from bottom and top)  
    - build a section of the image (trace) along the transition direction determined as above using nearest neighbour interpolation, the length of it is set by SECTION_LENGTH.
    - for the trace determined above, find min (SECTION_MIN) and max (SECTION_MAX) values. Check if the transition is "good" using two conditions:
        - SECTION_MIN < MIN_CRITERION (in other words the SECTION_MIN value is no more that 10% higher that the subset MIN value)
        - SECTION_MAX < MAX_CRITERION (in other words the SECTION_MAX value is no more that 10% lower that the subset MAX value)
    - if the transition is "good", find its 37% and 63% transition points by using nearest neighbour interpolation. Then find 37% to 63% transition distance.
'''
#
############################################


# sin function needs a y-offset -> c
def sin2x_fun(x,a,b,c):
    return a*np.sin(2.0*x+b)+c


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    #return wa*Ia + wb*Ib + wc*Ic + wd*Id
    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


def estimate_edge_transition(image, center, gradient, **kwargs):
    '''
    Estimate transition parameters given point (center, gradient directions) in the image. gleb.shtengel@gmail.com 06/2023
    
    Parameters:
    image : 2D array
        image array
    center : [int, int]
        Y, and X coordinates of the point for transition estimation
    gradient : [float, float]
        Y- and X- components of image gradient
    
    kwargs:
    bounds : list [low_bound, high_bound]
        list two values for estimating the transitions. If not defined, assumed to be [0.33, 0.67]
    subset_size : int
        subset size for min_criterion and max_criterion evaluation
        if not defined, 20 will be used
    thr_min_criterion  :  float
        minimum threshold value (from image CDF min) that transitition must reach to be counted IN
        if absent, 0.2 of image subset CDF level will be used
    thr_max_criterion  :  float
        maximum threshold value (from image CDF max) that transitition must reach to be counted IN
        if absent, 0.2 of image subset CDF level will be used
    section_length  :  float
        maximum distance from given center point allowed to reach min_criterion or max_criterion
        if absent, section_length=10 will be used
    min_max_aperture : int
        aperture size for estimating the min and max values for the transition estimation.
    transition_low_limit : float
        error flag is incremented by 4 if the determined transition distance is below this value. Default is 0.0
    transition_high_limit : float
        error flag is incremented by 8 if the determined transition distance is above this value. Default is 20.0
    disp_res : boolean
        display the results
    verbose : boolean
        print the outputs
    axs : ax artists
        if present, will plot to external artist. if not present, will execute fig, axs = sublots()
        
    Returns:
    error_flag : int
        0 - no error
        1 - failed to find min_criterion
        2 - failed to find max_criterion
        4 - if determined transition distance is below transition_low_limit
        8 - if determined transition distance is above transition_high_limit
        16 - if determined transition distance is nan
    following are returned non-zero if error_flag is non-zero
    transition_distance : float
        estimated transition length (from transition_min to transition_max)
    img_pix, img_val
        interpolated pixel coordinate and image value along the gradient line used for transition evaluation.        
    '''
    bounds = kwargs.get('bounds', [0.33, 0.67])
    disp_res = kwargs.get('disp_res', False)
    verbose = kwargs.get('verbose', False)
    subset_size = kwargs.get('subset_size', 20)
    section_length = kwargs.get('section_length', 20)
    thr_min_criterion = kwargs.get('thr_min_criterion', 0.2)
    thr_max_criterion = kwargs.get('thr_max_criterion', 0.2)
    min_max_aperture = kwargs.get('min_max_aperture', 5)
    transition_low_limit = kwargs.get('transition_low_limit', 0.0)
    transition_high_limit = kwargs.get('transition_high_limit', 10.0)
    Y, X = center
    ysz, xsz = image.shape
    Xi = np.max((0, X-subset_size//2))
    Xa = np.min((xsz-1, Xi+subset_size))
    Xi = Xa - subset_size
    Yi = np.max((0, Y-subset_size//2))
    Ya = min((ysz-1, Yi+subset_size))
    Yi = Ya - subset_size
    image_subset = image[Yi:Ya,Xi:Xa]

    min_criterion, max_criterion = get_min_max_thresholds(image_subset, thr_min=thr_min_criterion, thr_max=thr_max_criterion, disp_res=False )

    GrY, GrX = gradient
    cosY, cosX = gradient/np.sqrt(GrX*GrX+GrY*GrY)
    dist_pix = np.arange(-section_length//2+1,section_length//2+1)
    # center point is (section_length-1)//2
    x_positions = X + cosX*dist_pix
    y_positions = Y + cosY*dist_pix
    img_pix = (y_positions, x_positions)
    
    # image slice along the above coordinates : img_val
    img_val = bilinear_interpolate(image, x_positions, y_positions)
    jc = (section_length-1)//2
    if disp_res:
        axs = kwargs.get('axs', [False, False])
        if not axs[0]:
            fig, axs = plt.subplots(2,1, figsize=(6,6))
        vmin, vmax = get_min_max_thresholds(image_subset, disp_res=False )
        axs[0].imshow(image_subset, cmap='Greys')
        axs[0].axis(False)
        axs[0].plot(x_positions-Xi, y_positions-Yi, color='magenta')
        axs[0].plot(X-Xi, Y-Yi, 'gd')
        axs[1].plot(dist_pix, img_val, color='magenta', marker='o', markersize=2.0, label='interp. section')
        axs[1].plot(dist_pix[jc], img_val[jc], 'gd', label='center')
        axs[1].set_xlabel('Distance (pix)')
        axs[1].set_ylabel('Image Value')
        axs[1].grid(True)
        
    # check min and max
    error_flag = 0
    #
    # THAT IS WHAT I HAD INITIALLY - BUT THIS IS NOISY
    #section_min = np.min(img_val[0:(section_length-1)//2+1])
    #section_max = np.max(img_val[(section_length-1)//2:])
    #
    loc_min = np.argmin(img_val[0:(section_length-1)//2+1])
    loc_max = (section_length-1)//2 + np.argmax(img_val[(section_length-1)//2:])
    delta_pix = np.arange(-min_max_aperture//2+1,min_max_aperture//2+1)
    min_pix_ind = loc_min + delta_pix
    if np.min(min_pix_ind) < 0:
        min_pix_ind = min_pix_ind - np.min(min_pix_ind)
    min_pix = dist_pix[min_pix_ind]
    min_val_mean = np.mean(img_val[min_pix_ind])
    min_val = img_val[min_pix_ind]*0.0 + min_val_mean
    
    max_pix_ind = loc_max + delta_pix
    if np.max(max_pix_ind) > len(img_val)-1:
        max_pix_ind = max_pix_ind - (np.max(max_pix_ind)-len(img_val))-1
    max_pix = dist_pix[max_pix_ind]
    max_val_mean = np.mean(img_val[max_pix_ind])
    max_val = img_val[max_pix_ind]*0.0 + max_val_mean
   
    if min_val_mean > min_criterion:# + (max_criterion-min_criterion)/10.0:
        if verbose:
            print('Section_min = {:.2f}'.format(section_min))
            print('Min criterion = {:.2f}'.format(min_criterion))
        error_flag += 1
    if max_val_mean < max_criterion:# - (max_criterion-min_criterion)/10.0:
        if verbose:
            print('Section_max = {:.2f}'.format(section_max))
            print('Max criterion = {:.2f}'.format(max_criterion))
        error_flag += 2
    
    if verbose:
        print('Error Flag = {:d}'.format(error_flag))
    if error_flag == 0:
        loc_min = np.argmin(img_val[0:(section_length-1)//2+1])
        loc_max = (section_length-1)//2 + np.argmax(img_val[(section_length-1)//2:])
        if disp_res:
            axs[1].plot(dist_pix[loc_min], img_val[loc_min], 'bx', label='Min')
            axs[1].plot(dist_pix[loc_max], img_val[loc_max], 'rx', label='Max')
        delta_pix = np.arange(-min_max_aperture//2+1,min_max_aperture//2+1)
        
        min_pix_ind = loc_min + delta_pix
        if np.min(min_pix_ind) < 0:
            min_pix_ind = min_pix_ind - np.min(min_pix_ind)
        min_pix = dist_pix[min_pix_ind]
        min_val_mean = np.mean(img_val[min_pix_ind])
        min_val = img_val[min_pix_ind]*0.0 + min_val_mean
        
        max_pix_ind = loc_max + delta_pix
        if np.max(max_pix_ind) > len(img_val)-1:
            max_pix_ind = max_pix_ind - (np.max(max_pix_ind)-len(img_val))-1
        max_pix = dist_pix[max_pix_ind]
        max_val_mean = np.mean(img_val[max_pix_ind])
        max_val = img_val[max_pix_ind]*0.0 + max_val_mean
        
        transition_min = min_val_mean + bounds[0] * (max_val_mean - min_val_mean)
        transition_max = min_val_mean + bounds[1] * (max_val_mean - min_val_mean)

        if disp_res:
            axs[1].plot(min_pix, min_val, 'b', linestyle='dashed', label='Mean Min')
            axs[1].plot(max_pix, max_val, 'r', linestyle='dashed', label='Mean Max')
            
        #now calculate the transition
        # first, go up from center point
        
        j = jc-1
        while j < (len(img_val)-1) and img_val[j] < transition_max:
            j += 1
        ja = j
        xa = dist_pix[j-1] + (transition_max-img_val[j-1])*(dist_pix[j]-dist_pix[j-1])/(img_val[j]-img_val[j-1])
        # second, go down from center point
        j = jc+1
        while j > 0  and img_val[j] > transition_min:
            j -= 1
        ji = j
        xi = dist_pix[j+1] + (transition_min-img_val[j+1])*(dist_pix[j]-dist_pix[j+1])/(img_val[j]-img_val[j+1])
         
        transition_distance=np.abs((xa-xi))
        if verbose:
            print(dist_pix[0], transition_min-0.1*(max_val_mean - min_val_mean), '{:.2f}'.format(bounds[0]))
            print('Transition {:.2f} value = {:.2f}, Transition {:.2f} value = {:.2f}'.format(bounds[0], transition_min, bounds[1], transition_max))
            print('Transition {:.2f} position = {:.2f} pix, Transition {:.2f} position = {:.2f} pix'.format(bounds[0], xi, bounds[1], xa))
            print('Transition distance = {:.2f} pixels'.format(transition_distance))
        if disp_res:     
            axs[1].plot([dist_pix[0], dist_pix[-1]], [transition_min, transition_min], 'b', linewidth=0.5, label='Transition Min')   
            axs[1].plot([xi, xi], [min_val_mean, max_val_mean], 'b', linewidth=0.5)
            axs[1].plot([dist_pix[0], dist_pix[-1]], [transition_max, transition_max], 'r', linewidth=0.5, label='Transition Max')
            axs[1].plot([xa, xa], [min_val_mean, max_val_mean], 'r', linewidth=0.5)
            axs[1].text(dist_pix[0], transition_min-0.1*(max_val_mean - min_val_mean), '{:.2f}'.format(bounds[0]))
            axs[1].text(dist_pix[0], transition_max+0.05*(max_val_mean - min_val_mean), 'Min + {:.2f} * (Max-Min)'.format(bounds[1]))
            axs[1].text(dist_pix[0], transition_min+0.4*(transition_max - transition_min), 'Transition={:.2f} pix'.format(transition_distance))
            axs[1].plot(dist_pix[ji], img_val[ji], 'b+', label='Pt Past Targ. Mn')
            axs[1].plot(dist_pix[ja], img_val[ja], 'r+', label='Pt Past Targ. Mx')
            axs[1].legend(loc='right')

    else:
        transition_distance=0
    if transition_distance < transition_low_limit:
        error_flag += 4
    if transition_distance > transition_high_limit:
        error_flag += 8
    if np.isnan(transition_distance):
        error_flag += 16
        
    return error_flag, transition_distance, img_pix, img_val


def analyze_edge_transitions_image(image, **kwargs):
    '''
    Estimate transitions in the image. gleb.shtengel@gmail.com  06/2023
    
    Parameters:
    image : 2D array
        image array
        
    kwargs:
    edge_detector : string
        if 'Gradient' (default), absolute value of gradient will be used to select edge points
        another option is 'Canny'
    bounds : list [low_bound, high_bound]
        list two values for estimating the transitions. If not defined, assumed to be [0.33, 0.67]
    pixel_size : float
        pixel size in nm. Default is 1.0
    neighbour_exclusion_radius : int
        radius of exclusion (pixels). Transitions will be separated by at least this number of pixels. Default is 20.
    exclude_center : boolean
        if Yes, the center circle of center_exclusion_radius below will be exculded (center mark may be there). Default is True
    center_exclusion_radius : int
        exclusion radius for the center (see above). . Default is 20.
    subset_size : int
        subset size (pixels) for min_criterion and max_criterion evaluation
        Default is 50.
    section_length  :  float
        the length of the cross-section profile (in poixels), centered at a given center point,
        along the direction of highest local gradient, used for transition evaluation. Default is 50.
    kernel : 2D float array
        a kernel to perform 2D smoothing convolution. default is np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]), where st=1/sqrt(2)
    perform_smoothing : boolean
        perform smoothing prior to edge detection. used for finding the edge point only, is NOT used during transition analysis.
        default is False.
    thr_min_criterion  :  float
        minimum threshold value (from image_subset CDF min) that cross-section profile must reach
        in order for this profile to be considered for transition evaluation. Defaults is 0.2 of image_subset CDF level
    thr_max_criterion  :  float
        maximum threshold value (from image_subset CDF max) that cross-section profile must reach
        in order for this profile to be considered for transition evaluation. Defaults is 0.2 of image_subset CDF level
    grad_thr : float
        threshold for selecting the transition points if Gradient Edge Detector is used.
        Default value id 0.005. Only the points with absolute value of gradient within the top fraction of CDG will be considered.
        For default value of 0.005 that means that only the points with absolute value of gradient within 0.005 of
        the max absolute of gradient will be considered.
    min_max_aperture : int
        aperture size for estimating the min and max values for the transition estimation.
    transition_low_limit : float
        min value allowed for the transition (pixels). Default is 0.0.
        error flag is incremented by 4 if the determined transition distance is below this value.
    transition_high_limit : float
        min value allowed for the transition (pixels). Default is 10.0.
        error flag is incremented by 8 if the determined transition distance is above this value.
    verbose : boolean
        print the outputs
    results_file_xlsx : file name for Excel workbook to save the results
       
    Returns:
    results_file_xlsx, centers_selected, center_grads_selected, transition_distances_selected
    results_file_xlsx : file name of the Excel workbook with the results
    centers_selected : array of center coordinates (Y, X) of analyzed transitions
    center_grads_selected : array of gradients at the center points of analyzed transitions
    transition_distances_selected : array of calculated transion distances
    '''
    edge_detector = kwargs.get('edge_detector', 'Gradient')
    bounds = kwargs.get('bounds', [0.33, 0.67])
    pixel_size = kwargs.get('pixel_size', 1.0)
    neighbour_exclusion_radius = kwargs.get('neighbour_exclusion_radius', 20)
    exclude_center = kwargs.get('exclude_center', True)
    center_exclusion_radius = kwargs.get('center_exclusion_radius', 20)
    subset_size = kwargs.get('subset_size', 50)
    section_length = kwargs.get('section_length', 50)
    thr_min_criterion =  kwargs.get('thr_min_criterion', 0.2)
    thr_max_criterion =  kwargs.get('thr_max_criterion', 0.2)
    canny_threshold1 = kwargs.get('canny_threshold1', 70)
    canny_threshold2 = kwargs.get('canny_threshold2', 70)
    canny_apertureSize = kwargs.get('canny_apertureSize', 3)
    canny_L2gradient = kwargs.get('canny_L2gradient', True)

    grad_thr = kwargs.get('grad_thr', 0.005)
    min_max_aperture = kwargs.get('min_max_aperture', 5)
    transition_low_limit = kwargs.get('transition_low_limit', 0.0)
    transition_high_limit = kwargs.get('transition_high_limit', 10.0)
    verbose = kwargs.get('verbose', False)
    results_file_xlsx = kwargs.get('results_file_xlsx', 'results.xlsx')

    st = 1.0/np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
    def_kernel = def_kernel/def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    perform_smoothing = kwargs.get('perform_smoothing', False)

    kwargs['edge_detector'] = edge_detector
    kwargs['bounds'] = bounds
    kwargs['pixel_size'] = pixel_size
    kwargs['neighbour_exclusion_radius'] = neighbour_exclusion_radius
    kwargs['exclude_center'] = exclude_center
    kwargs['center_exclusion_radius'] = center_exclusion_radius
    kwargs['subset_size'] = subset_size
    kwargs['section_length'] = section_length
    kwargs['thr_min_criterion'] = thr_min_criterion
    kwargs['thr_max_criterion'] = thr_max_criterion
    kwargs['canny_threshold1'] = canny_threshold1
    kwargs['canny_threshold2'] = canny_threshold2
    kwargs['canny_apertureSize'] = canny_apertureSize
    kwargs['canny_L2gradient'] = canny_L2gradient
    kwargs['grad_thr'] = grad_thr
    kwargs['min_max_aperture'] = min_max_aperture
    kwargs['transition_low_limit'] = transition_low_limit
    kwargs['transition_high_limit'] = transition_high_limit
    kwargs['verbose'] = verbose
    kwargs['perform_smoothing'] = perform_smoothing
    kwargs['kernel'] = kernel
    kwargs['results_file_xlsx'] = results_file_xlsx

    ysz, xsz = image.shape
    xind, yind = np.meshgrid(np.arange(xsz), np.arange(ysz), sparse=False)
    
    if perform_smoothing:
        grad = np.gradient(convolve2d(image, kernel, mode='same'))
    else:
        grad = np.gradient(image)
    grad_array = np.array(grad)
    abs_grad = np.sqrt(grad[0]*grad[0]+grad[1]*grad[1])
    grad_min, grad_max = get_min_max_thresholds(abs_grad, thr_min=0.005, thr_max=grad_thr, disp_res=False)
    
    if edge_detector == 'Canny':
        if perform_smoothing:
            image0 = convolve2d(image, kernel, mode='same')
        else:
            image0 = image
        if type(image[0,0]) == np.uint8:
            #print('8-bit image already - no need to convert')
            image_I8 = image0
        else:
            data_min, data_max = get_min_max_thresholds(image0, thr_min=1e-3, thr_max=1e-3, nbins=256, disp_res=False)
            image_I8 = ((np.clip(image0, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        canny_edges = cv2.Canny(image_I8, canny_threshold1, canny_threshold2,  apertureSize=canny_apertureSize, L2gradient = canny_L2gradient)
        cond = canny_edges == 255
    else:
        cond = abs_grad > grad_max

    cond[0:subset_size//2, :] = 0
    cond[-subset_size//2:, :] = 0
    cond[:, 0:subset_size//2] = 0
    cond[:, -subset_size//2:] = 0
    if verbose:
        print('Using '+edge_detector+' Edge Detector')
    
    grad_pts = abs_grad[cond]
    if verbose:
        print('Initial edge points count:',len(grad_pts))
    xpts = xind[cond]
    ypts = yind[cond]

    grad_inds = np.flip(np.argsort(grad_pts))
    grad_pts_sorted = np.flip(np.sort(grad_pts))
    xpts_sorted = np.array(xpts)[grad_inds]
    ypts_sorted = np.array(ypts)[grad_inds]
    
    xcenter = xsz//2
    ycenter = ysz//2
    if exclude_center:
        dist_arr = np.sqrt((xpts_sorted-xcenter)*(xpts_sorted-xcenter)+(ypts_sorted-ycenter)*(ypts_sorted-ycenter))
        grad_pts_sorted = grad_pts_sorted[dist_arr>center_exclusion_radius]
        xpts_sorted = xpts_sorted[dist_arr>center_exclusion_radius]
        ypts_sorted = ypts_sorted[dist_arr>center_exclusion_radius]

    centers = []
    center_grads = []
    while len(grad_pts_sorted) > 0:
        # first - add a peak to collection
        xpeak = xpts_sorted[0]
        ypeak = ypts_sorted[0]
        centers.append([ypeak, xpeak])
        center_grads.append(np.mean(np.mean(grad_array[:, ypeak-3:ypeak+3, xpeak-3:xpeak+3], axis=2), axis=1))
        # now remove the points within the excusion circle from the consideration
        dist_arr = np.sqrt((xpts_sorted-xpeak)*(xpts_sorted-xpeak)+(ypts_sorted-ypeak)*(ypts_sorted-ypeak))
        grad_pts_sorted = grad_pts_sorted[dist_arr>neighbour_exclusion_radius]
        xpts_sorted = xpts_sorted[dist_arr>neighbour_exclusion_radius]
        ypts_sorted = ypts_sorted[dist_arr>neighbour_exclusion_radius]
    if verbose:
        print('Verified edge points: ', len(centers))
    
    centers = np.array(centers)
    center_grads = np.array(center_grads)
    Y = centers[:,0]
    X = centers[:,1]

    results = []
    img_vals_all = []
    kwargs_loc = {'bounds' : bounds,
                  'pixel_size' : pixel_size,
                  'disp_res' : False,
                  'verbose' : verbose,
                  'subset_size' : subset_size,
                  'section_length' : section_length,
                  'thr_min_criterion' : thr_min_criterion,
                  'thr_max_criterion' : thr_max_criterion,
                  'min_max_aperture' : min_max_aperture,
                  'transition_low_limit' : transition_low_limit,
                  'transition_high_limit' : transition_high_limit,
                  'verbose': False,
                  'disp_res' : False}
    for center, center_grad in zip(tqdm(centers, desc='analyzing transitions', display=verbose), center_grads):
        res = estimate_edge_transition(image, center, center_grad,
                                  **kwargs_loc)
        results.append(res[0:2])
        img_vals_all.append(res[3])
    if verbose:
        print('Analyzed transtions: ', len(results))
    results = np.array(results)
    img_vals_all = np.array(img_vals_all)
    error_flags = results[:, 0]
    transition_distances = results[:, 1]
    if verbose:
        print('Good transtions:     ', len(error_flags[error_flags==0]))

    transition_distances_selected = np.array(transition_distances[error_flags==0]).astype(float)
    X_selected = X[error_flags==0]
    Y_selected = Y[error_flags==0]
    X_grads = center_grads[:,1]
    Y_grads = center_grads[:,0]
    X_grads_selected = X_grads[error_flags==0]
    Y_grads_selected = Y_grads[error_flags==0]
    centers_selected = centers[error_flags==0]
    center_grads_selected = center_grads[error_flags==0]
    cosXs = X_grads/np.sqrt(X_grads*X_grads+Y_grads*Y_grads)
    cosYs = Y_grads/np.sqrt(X_grads*X_grads+Y_grads*Y_grads)
    cosXs_selected = cosXs[error_flags==0]
    cosYs_selected = cosYs[error_flags==0]
    tr_mean = np.mean(transition_distances_selected)
    tr_std = np.std(transition_distances_selected)
    
    theta = np.array(np.angle(cosXs_selected+1.0j*cosYs_selected)).astype(float)
    # sin2x_fun
    #a*np.sin(.0*x+b)+c
    theta_ordered = np.sort(theta)
    try:
        p_opt, p_cov = cf(sin2x_fun, theta, transition_distances_selected, p0=(tr_std, 0.0, tr_mean))
    except:
        p_opt = [1.0, 1.0, 1.0]

    if verbose:
        print('Saving the results into the file: ', results_file_xlsx)
    xlsx_writer = pd.ExcelWriter(results_file_xlsx, engine='xlsxwriter')
    trans_str = '{:.2f} to {:.2f} trasntition (pix)'.format(bounds[0], bounds[1])
    columns=['X', 'Y', 'X grad', 'Y grad', 'error_flag', trans_str]
    res_summary = pd.DataFrame(np.vstack((X, Y, X_grads, Y_grads, error_flags, transition_distances)).T, columns = columns, index = None)
    res_summary.to_excel(xlsx_writer, index=None, sheet_name='Transition analysis results')
    img_vals_pd = pd.DataFrame(img_vals_all)
    img_vals_pd.to_excel(xlsx_writer, index=None, sheet_name='Transitions Intensity Profiles')
    kwargs['Transition Mean (pix)'] = tr_mean
    kwargs['Transition STD (pix)'] = tr_std
    kwargs['Assymetry'] = np.abs(p_opt[0]/p_opt[2])
    kwargs_info = pd.DataFrame([kwargs]).T # prepare to be save in transposed format
    kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')
    #xlsx_writer.save()
    xlsx_writer.close()
        
    return results_file_xlsx, centers_selected, center_grads_selected, transition_distances_selected



################################################################
#
#   helper functions to plot the results: BLOB Transitions
#
################################################################

def plot_blob_map_and_results_single_image(image, results_xlsx, **kwargs):
    '''
    Plots the map of extracted blobs over the image. gleb.shtengel@gmail.com  06/2023
    
    Parameters:
    ----------
    image : 2D array
        image array

    results_file_xlsx : file name for Excel workbook with the blob data
        
    kwargs:
    ----------
    nbins : int
        Number of bins for histogram. Default is 64.
    save_png : boolean
        Save the image into PNG file. Default is False.
    save_fname : string
        File name for to save the image. Default is created by results_xlsx.replace('.xlsx', '_map.png').
    dpi : int
        DPI, default is 300.

    Returns:
    ----------
    save_fname : str

    '''

    save_png = kwargs.get('save_png', False)
    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    pixel_size = saved_kwargs.get("pixel_size", 0.0)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_map.png'))
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    nbins = kwargs.get('nbins', 64)
    dpi = kwargs.get('dpi', 300)
    
    xs=7.0
    ysz, xsz = image.shape
    ys = xs*ysz/xsz
    text_col = 'brown'
    text_fs = 12
    axis_label_fs = 10
    table_fs = 12
    caption_fs = 8
    
    trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
    columns=['X', 'Y', 'R', 'error_flag',
             trans_str + ' X-pt1',
             trans_str + ' X-pt2',
             trans_str + ' Y-pt1',
             trans_str + ' Y-pt2',
             trans_str + ' X-slp1',
             trans_str + ' X-slp2',
             trans_str + ' Y-slp1',
             trans_str + ' Y-slp2']

    int_results = pd.read_excel(results_xlsx, sheet_name='Transition analysis results')
    error_flags = int_results['error_flag']
    Xpt1 = np.array(int_results[trans_str + ' X-pt1'])[error_flags==0]
    Xpt2 = np.array(int_results[trans_str + ' X-pt2'])[error_flags==0]
    Ypt1 = np.array(int_results[trans_str + ' Y-pt1'])[error_flags==0]
    Ypt2 = np.array(int_results[trans_str + ' Y-pt2'])[error_flags==0]
    Xslp1 = np.array(int_results[trans_str + ' X-slp1'])[error_flags==0]
    Xslp2 = np.array(int_results[trans_str + ' X-slp2'])[error_flags==0]
    Yslp1 = np.array(int_results[trans_str + ' Y-slp1'])[error_flags==0]
    Yslp2 = np.array(int_results[trans_str + ' Y-slp2'])[error_flags==0]
    X = int_results['X']
    Y = int_results['Y']
    X_selected = X[error_flags==0]
    Y_selected = Y[error_flags==0]
    X_unselected = X[error_flags>0]
    Y_unselected = Y[error_flags>0]

    XYpt_selected = [Xpt1, Xpt2, Ypt1, Ypt2]
    Xpt_selected = [Xpt1, Xpt2]
    Ypt_selected = [Ypt1, Ypt2]
    Xslp_selected = [Xslp1, Xslp2]
    Yslp_selected = [Yslp1, Yslp2]
    tr_mean = np.mean(XYpt_selected)
    tr_std = np.std(XYpt_selected)
    
    fig, axs = plt.subplots(1, 2, figsize=(2*xs,ys))
    ax = axs[0]
    ax.imshow(image, cmap='Greys')
    ax.scatter(X_unselected, Y_unselected, facecolors='none', color='blue', marker = 'o', s=10, linewidth=0.5)
    ax.scatter(X_selected, Y_selected, facecolors='none', color='lime', marker = 'o', s=10, linewidth=0.5)
    ax.scatter(X_selected[0:3], Y_selected[0:3], color='red', facecolors='none', marker = 'o', s=20, linewidth=0.75)
    ax.scatter(X_selected[-3:], Y_selected[-3:], color='cyan', facecolors='none', marker = 'o', s=20, linewidth=0.75)
    ax.axis(False)
    ax.text(0.15, 0.97, 'Blobs determined by Laplasian of Gaussians', transform=ax.transAxes, fontsize=text_fs)
    ax.text(0.07, 0.90, '# of blobs: {:d}'.format(len(X_selected)), transform=ax.transAxes, color=text_col, fontsize=text_fs)
    ax.text(0.07, 0.85, '{:.0f}% - {:.0f}% Transitions'.format(bounds[0]*100, bounds[1]*100), transform=ax.transAxes, color=text_col, fontsize=text_fs)
    ax.text(0.07, 0.80, 'Pixel Size (nm): {:.3f}'.format(pixel_size), transform=ax.transAxes, color=text_col, fontsize=text_fs)
    ax.text(0.07, 0.75, 'Mean value (nm): {:.3f}'.format(tr_mean), transform=ax.transAxes, color=text_col, fontsize=text_fs)
    ax.text(0.07, 0.70, 'STD (nm):       {:.3f}'.format(tr_std), transform=ax.transAxes, color=text_col, fontsize=text_fs)
    axs[1].axis(False)
    ax1 = plt.subplot(2,2,2)
    ax2 = plt.subplot(2,2,4)
    
    tr_sets = [[Xpt_selected, Ypt_selected],
          [Xslp_selected, Yslp_selected]]
    fexts =['_{:.0f}{:.0f}pts'.format(bounds[0]*100, bounds[1]*100), '_{:.0f}{:.0f}slp'.format(bounds[0]*100, bounds[1]*100)]
    xaxis_labels = ['{:.0f}%-{:.0f}% Transition (pts) (nm)'.format(bounds[0]*100, bounds[1]*100),
        '{:.0f}%-{:.0f}% Transition (slope) (nm)'.format(bounds[0]*100, bounds[1]*100)]
    hranges = [(0, 10.0), 
           (0, 10.0)]  # histogram range for the transition distance (in nm))
       
    for [tr_xs, tr_ys], fext, xaxis_label, hrange, axloc in zip(tr_sets, fexts, xaxis_labels, hranges,  [ax1, ax2]):
        trs = np.squeeze(np.array((tr_xs, tr_ys)).flatten())
        tr_x = np.array(tr_xs).flatten()
        tr_y = np.array(tr_ys).flatten()
        dsets = [trs, tr_x, tr_y]
        cols = ['blue', 'green', 'red']
        hst_data =  np.zeros((len(dsets), 4))
        cc_data =  np.zeros((nbins, len(dsets)+1))
        for (j, dset), col in zip(enumerate(dsets), cols):
            hst_res = np.array(add_hist(dset, nbins=nbins, hrange=hrange, ax=axloc, col=col, label=''), dtype=object)
            hst_data[j, :] = hst_res[2:7]
            #full_hst_data[j+1, :] = hst_data[j, :]
            cc_data[:, j+1] = hst_res[1]
        axloc.grid(True)
        axloc.set_xlim(hrange)
        axloc.set_xlabel(xaxis_label, fontsize=axis_label_fs)
        axloc.set_ylabel('Count', fontsize=axis_label_fs)
        plt.tick_params(labelsize = axis_label_fs)
        cond_bl =  'Point Analysis, {:d} blobs'.format(len(tr_x)//2)

        columns = ['Hist. Peak', 'Median', 'Mean', 'STD']
        rows = ['X, Y', 'X', 'Y']
        n_cols = len(columns)
        n_rows = len(rows)
        cell_text = [['{:.2f}'.format(d) for d in dd] for dd in hst_data]

        tbl = axloc.table(cellText=cell_text,
                         rowLabels=rows,
                         colLabels=columns,
                         colWidths=[0.14]*n_cols,
                         cellLoc='center',
                         colLoc='center',
                         loc=1,
                         zorder=10)

        tbl.scale(1.0, 1.5)
        table_props = tbl.properties()
        table_cells = table_props['child_artists']

        for j, cell in enumerate(table_cells[0:n_cols*n_rows]):
            cell.get_text().set_color(cols[j//n_cols])
            cell.get_text().set_fontsize(table_fs)
        for j, cell in enumerate(table_cells[n_cols*(n_rows+1):]):
            cell.get_text().set_color(cols[j])
        for cell in table_cells[n_cols*n_rows:]:
        #    cell.get_text().set_fontweight('bold')
            cell.get_text().set_fontsize(table_fs)
    
    fig.subplots_adjust(left=0.01, bottom=0.06, right=0.99, top=0.98, wspace=0.02, hspace=0.20)
    if save_png:
        ax.text(0.05, -0.05, save_fname, transform=ax.transAxes, fontsize=caption_fs)
        fig.savefig(save_fname, dpi=dpi)
    else:
        save_fname = 'Figure not saved'

    return save_fname


def plot_blob_examples_single_image(image, results_xlsx, **kwargs):
    save_png = kwargs.get('save_png', False)
    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    pixel_size = saved_kwargs.get("pixel_size", 1.0)
    subset_size = saved_kwargs.get("subset_size", 2.0)
    dx2 = subset_size//2
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_examples.png'))
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    bands = saved_kwargs.get("bands", [3, 2, 3])
    verbose = kwargs.get('verbose', False)

    xs=16.0
    ysz, xsz = image.shape
    ys = xs*5.0/4.0
    text_col = 'brown'
    fst = 40
    fs = 12
    fs_legend = 10
    fs_labels = 12
    caption_fs = 8
    
    trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
    int_results = pd.read_excel(results_xlsx, sheet_name='Transition analysis results')
    error_flags = int_results['error_flag']

    X = int_results['X']
    Y = int_results['Y']
    X_selected = X[error_flags==0]
    Y_selected = Y[error_flags==0]

    Xs = np.concatenate((X_selected[0:3], X_selected[-3:]))
    Ys = np.concatenate((Y_selected[0:3], Y_selected[-3:]))
    
    xt = 0.0
    yt=1.5
    
    clr_x = 'green'
    clr_y = 'blue'
    fig, axs = plt.subplots(4,3, figsize=(xs, ys))
    fig.subplots_adjust(left=0.02, bottom=0.04, right=0.99, top=0.99, wspace=0.15, hspace=0.12)

    ax_maps = [axs[0,0], axs[0,1], axs[0,2], axs[2,0], axs[2,1], axs[2,2]]
    ax_profiles = [axs[1,0], axs[1,1], axs[1,2], axs[3,0], axs[3,1], axs[3,2]]

    for j, x in enumerate(Xs):
        y = Ys[j]
        xx = int(x)
        yy = int(y)
        subset = image[yy-dx2:yy+dx2, xx-dx2:xx+dx2]
        #print(np.mean(subset))
        ax_maps[j].imshow(subset, cmap='Greys')#, vmin=0, vmax=160)
        ax_maps[j].grid(False)
        ax_maps[j].axis(False)
        crop_x = patches.Rectangle((-0.5,dx2-0.5),subset_size,1, linewidth=1, edgecolor=clr_x , facecolor='none')
        crop_y = patches.Rectangle((dx2-0.5, -0.5),1,subset_size, linewidth=1, edgecolor=clr_y, facecolor='none')
        ax_maps[j].add_patch(crop_x)
        ax_maps[j].add_patch(crop_y)
        ax_maps[j].text(xt,yt,'X-Y', color='black',  bbox=dict(facecolor='white', edgecolor='none'), fontsize=fst)
        amp_x = subset[dx2, :]
        amp_y = subset[:, dx2]

        a0 = np.min(np.array((amp_x, amp_y)))
        a1 = np.max(np.array((amp_x, amp_y)))
        amp_scale = (a0-(a1-a0)/10.0, a1+(a1-a0)/10.0)
        #print(amp_scale)
        #print(shape(amp_x), shape(amp_y), shape(amp_z))
        tr_x = analyze_blob_transitions(amp_x, pixel_size=pixel_size,
                                col = clr_x,
                                cols=['green', 'green', 'black'],
                                bounds=bounds,
                                bands = bands,
                                y_scale = amp_scale,
                                disp_res=True, ax=ax_profiles[j], pref = 'X-',
                                fs_labels = fs_labels, fs_legend = fs_legend,
                                verbose = verbose)
        ax_profiles[j].legend(loc='upper left', fancybox=False, edgecolor="w", fontsize = fs_legend)

        ax2 = ax_profiles[j].twinx()
        tr_y = analyze_blob_transitions(amp_y, pixel_size=pixel_size,
                                col = clr_y,
                                cols=['blue', 'blue', 'black'],
                                bounds=bounds,
                                bands = bands,
                                y_scale = amp_scale,
                                disp_res=True, ax=ax2, pref = 'Y-',
                                fs_labels = fs_labels, fs_legend = fs_legend,
                                       verbose = verbose)
        ax2.legend(loc='upper right', fancybox=False, edgecolor="w", fontsize = fs_legend)
        ax_profiles[j].tick_params(labelleft=False)
        ax2.tick_params(labelright=False)
        ax2.get_yaxis().set_visible(False)

    if save_png:
        axs[3,0].text(0.00, -0.15, save_fname, transform=axs[3,0].transAxes, fontsize=caption_fs)
        fig.savefig(save_fname, dpi=300)



################################################################
#
#   helper functions to plot the results: Edge Transitions
#
################################################################

def plot_edge_transition_analysis_details(image, results_xlsx, **kwargs):
    '''
    Plots the map of extracted edge points over the image and the analysis results. gleb.shtengel@gmail.com  06/2023
    
    Parameters:
    ----------
    image : 2D array
        image array

    results_file_xlsx : file name for Excel workbook with the blob data
        
    kwargs:
    ----------
    nbins : int
        Number of bins for histogram. Default is 64.
    color_index : str
        Determins what parameter is used to color-code the detected transitions and transition plots.
        Options are: 'Tr' - transition distance value (default), 'X' - x-coordinate of the transition point, 'Y' - y-coordinate of the transition point.
    save_png : boolean
        Save the image into PNG file. Default is False.
    save_fname : string
        File name for to save the image. Default is created by results_xlsx.replace('.xlsx', '_map.png').
    fontsize : int
        Font size for annotations. Default is 12.
    dpi : int
        DPI, default is 300.

    Returns:
    ----------
    save_fname : str
    '''

    save_png = kwargs.get('save_png', False)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_analysis.png'))
    color_index = kwargs.get('color_index', 'Tr')
    fontsize = kwargs.get('fontsize', 12)
    dpi = kwargs.get('dpi', 300)

    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    edge_detector = saved_kwargs.get("edge_detector", '')
    section_length = saved_kwargs.get("section_length", 50)
    pixel_size = saved_kwargs.get("pixel_size", 0.0)
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])

    int_results = pd.read_excel(results_xlsx, sheet_name='Transition analysis results')
    X = int_results['X']
    Y = int_results['Y']
    X_grads = int_results['X grad']
    Y_grads = int_results['Y grad']
    error_flags = int_results['error_flag']
    X_selected = np.array(X[error_flags==0])
    Y_selected = np.array(Y[error_flags==0])
    ntr = len(X_selected)
    X_grads_selected = np.array(X_grads[error_flags==0])
    Y_grads_selected = np.array(Y_grads[error_flags==0])
    cosXs_selected = X_grads_selected/np.sqrt(X_grads_selected*X_grads_selected+Y_grads_selected*Y_grads_selected)
    cosYs_selected = Y_grads_selected/np.sqrt(X_grads_selected*X_grads_selected+Y_grads_selected*Y_grads_selected)
    
    trans_str = '{:.2f} to {:.2f} trasntition (pix)'.format(bounds[0], bounds[1])
    transition_distances = int_results[trans_str] 
    transition_distances_selected = np.array(transition_distances[error_flags==0]).astype(float)
    tr_mean = np.mean(transition_distances_selected)
    tr_std = np.std(transition_distances_selected)

    if color_index == 'X':
        sorted_indices = np.argsort(X_selected).tolist()
    elif color_index == 'Y':
        sorted_indices = np.argsort(Y_selected).tolist()
    else:
        sorted_indices = np.argsort(transition_distances_selected).tolist()

    transition_distances_selected = transition_distances_selected[sorted_indices]
    X_selected = X_selected[sorted_indices]
    Y_selected = Y_selected[sorted_indices]
    cosXs_selected = cosXs_selected[sorted_indices]
    cosYs_selected = cosYs_selected[sorted_indices]
    
    tr_x_to_plot = cosXs_selected*transition_distances_selected
    tr_y_to_plot = cosYs_selected*transition_distances_selected
    tr_plot_min = np.min((tr_x_to_plot, tr_y_to_plot))
    tr_plot_max = np.max((tr_x_to_plot, tr_y_to_plot))
    
    img_vals_all = np.array(pd.read_excel(results_xlsx, sheet_name='Transitions Intensity Profiles'))
    theta = np.array(np.angle(cosXs_selected+1.0j*cosYs_selected)).astype(float)
    theta_ordered = np.sort(theta)

    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.90, wspace=0.10, hspace=0.20)
    axs[0,0].imshow(image, cmap='Greys_r')
    dist_pix = np.arange(-section_length//2+1,section_length//2+1)
    for (j, X), Y, cosX, cosY, image_vals in zip(enumerate(X_selected), Y_selected, cosXs_selected, cosYs_selected, img_vals_all):
        x_positions = X + cosX*dist_pix
        y_positions = Y + cosY*dist_pix
        inds = np.where(np.logical_and(np.logical_and(x_positions>=0, x_positions<=xsz), np.logical_and(y_positions>=0, y_positions<=ysz)))
        my_col = plt.get_cmap("gist_rainbow_r")((ntr-j)/ntr)
        axs[0,0].plot(X, Y, linestyle='none', color=my_col, marker = 'o', markersize=2)
        axs[0,0].plot(x_positions[inds], y_positions[inds], 'yellow', linewidth=0.5)
        axs[1,0].plot(dist_pix, image_vals, color=my_col, linewidth=0.5)
        axs[1,1].plot(tr_x_to_plot[j], tr_y_to_plot[j], color=my_col, marker = 'o', markersize=2)
    axs[0,0].axis(False)
    axs[0,0].set_title('Image with Edge points')
    axs[0,1].text(0.5, 1.1, top_text, transform=axs[0,0].transAxes)
    axs[1,0].grid(True)
    axs[1,0].set_xlabel('Distance (nm)')
    axs[1,0].set_ylabel('Interpolated Image Intensity')
    axs[1,0].set_title('Analyzed Transitions: Intensity Profiles')

    hist_res = axs[0,1].hist(transition_distances_selected, bins=64)
    axs[0,1].grid(True)

    axs[0,1].text(0.05, 0.95, 'Number of Analyzed Transitions: {:d}'.format(len(transition_distances_selected)), transform=axs[0,1].transAxes)
    axs[0,1].text(0.05, 0.9, 'Mean value (pix): {:.3f}'.format(tr_mean), transform=axs[0,1].transAxes)
    axs[0,1].text(0.05, 0.85, 'STD (pix)  :        {:.3f}'.format(tr_std), transform=axs[0,1].transAxes)
    axs[0,1].text(0.05, 0.75, 'Pixel Size (nm): {:.3f}'.format(pixel_size), transform=axs[0,1].transAxes)
    axs[0,1].text(0.05, 0.7, 'Mean value (nm): {:.3f}'.format(tr_mean*pixel_size), transform=axs[0,1].transAxes)
    axs[0,1].text(0.05, 0.65, 'STD (nm):       {:.3f}'.format(tr_std*pixel_size), transform=axs[0,1].transAxes)
    tr_str = '{:.2f} to {:.2f} '.format(bounds[0], bounds[1])
    axs[0,1].set_title(tr_str + ' Transition Distribution ('+ edge_detector + ' Detector)', fontsize = fontsize)
    axs[0,1].set_xlabel('Transition Distance (pix)', fontsize = fontsize)
    axs[0,1].set_ylabel('Count', fontsize = fontsize)
    
    axs[1,0].axis('scaled')
    axs[1,0].grid(True)
    axs[1,0].set_title('Transition Distribution over Directions', fontsize = fontsize)
    axs[1,0].set_xlabel('Transition X component', fontsize = fontsize)
    axs[1,0].set_ylabel('Transition Y component', fontsize = fontsize)
   
    try:
        # sin2x_fun
        #a*np.sin(2.0*x+b)+c
        p_opt, p_cov=cf(sin2x_fun, theta, np.array(transition_distances_selected), p0=(tr_std, 0.0, tr_mean))
        tr_fit = sin2x_fun(theta_ordered, *p_opt)
        trX_fit = tr_fit*np.cos(theta_ordered)
        trY_fit = tr_fit*np.sin(theta_ordered)
        axs[1,0].plot(trX_fit, trY_fit, c='magenta', label='centered fit')
        axs[1,0].text(0.025, 0.95, 'Ellipticity: {:.3f}'.format(np.abs(p_opt[0]/p_opt[2])), transform=axs[1,0].transAxes, color='magenta')
    except:
        axs[1,0].text(0.025, 0.95, 'Could not analyze Ellipticity', transform=axs[1,0].transAxes, color='magenta')

    if save_png:
        axs[1,0].text(0.05, -0.18, save_fname, transform=axs[1,0].transAxes, fontsize = fontsize-2)
        fig.savefig(save_fname, dpi=dpi)


def plot_edge_transition_points_map(image, results_xlsx, **kwargs):
    save_png = kwargs.get('save_png', False)
    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    edge_detector = saved_kwargs.get("edge_detector", '')
    pixel_size = saved_kwargs.get("pixel_size", 0.0)
    section_length = saved_kwargs.get("section_length", 50)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_transition_map.png'))
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    assymetry = saved_kwargs.get("Assymetry", 0.0)

    int_results = pd.read_excel(results_xlsx, sheet_name='Transition analysis results')
    X = int_results['X']
    Y = int_results['Y']
    X_grads = int_results['X grad']
    Y_grads = int_results['Y grad']
    error_flags = int_results['error_flag']
    X_selected = X[error_flags==0]
    Y_selected = Y[error_flags==0]
    X_grads_selected = X_grads[error_flags==0]
    Y_grads_selected = Y_grads[error_flags==0]
    trans_str = '{:.2f} to {:.2f} trasntition (pix)'.format(bounds[0], bounds[1])
    transition_distances = int_results[trans_str] 
    transition_distances_selected = transition_distances[error_flags==0]
    tr_mean = np.mean(transition_distances_selected)
    tr_std = np.std(transition_distances_selected)
    
    xs=7.0
    ysz, xsz = image.shape
    ys = xs*ysz/xsz
    col = 'yellow'
    fs=12
    fig, ax = plt.subplots(1, 1, figsize=(xs,ys))
    fig.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.92, wspace=0.05, hspace=0.1)
    ax.imshow(image, cmap='Greys_r')
    dist_pix = np.arange(-section_length//2+1,section_length//2+1)
    for X, Y, GrX, GrY in zip(X_selected, Y_selected, X_grads_selected, Y_grads_selected):
        cosX = GrX/np.sqrt(GrX*GrX+GrY*GrY)
        cosY = GrY/np.sqrt(GrX*GrX+GrY*GrY)
        # center point is (section_length-1)//2
        x_positions = X + cosX*dist_pix
        y_positions = Y + cosY*dist_pix
        inds = np.where(np.logical_and(np.logical_and(x_positions>=0, x_positions<=xsz), np.logical_and(y_positions>=0, y_positions<=ysz)))
        ax.plot(x_positions[inds], y_positions[inds], 'r', linewidth=0.15)
    ax.plot(X_selected, Y_selected, linestyle='none', color='lime', marker = 'o', markersize=1)
    ax.scatter(X_selected[0:3], Y_selected[0:3], color='red', facecolors='none', marker = 'o', s=20)
    ax.scatter(X_selected[-4:-1], Y_selected[-4:-1], color='white', facecolors='none', marker = 'o', s=20)
    ax.axis(False)
    ax.set_title('Image with Edge points determined by ' + edge_detector + ' Edge Detector')
    ax.text(0.02, 0.95, '# of Transitions: {:d}'.format(len(transition_distances_selected)), transform=ax.transAxes, color=col, fontsize=fs)
    ax.text(0.02, 0.90, 'Pixel Size (nm): {:.3f}'.format(pixel_size), transform=ax.transAxes, color=col, fontsize=fs)
    ax.text(0.02, 0.85, 'Mean value (nm): {:.3f}'.format(tr_mean*pixel_size), transform=ax.transAxes, color=col, fontsize=fs)
    ax.text(0.02, 0.80, 'STD (nm):       {:.3f}'.format(tr_std*pixel_size), transform=ax.transAxes, color=col, fontsize=fs)
    ax.text(0.02, 0.75, 'Assymetry:    {:.3f}'.format(assymetry), transform=ax.transAxes, color=col, fontsize=fs)
    if save_png:
        ax.text(0.00, -0.03, save_fname, transform=ax.transAxes, fontsize=8)
        fig.savefig(save_fname, dpi=300)


def plot_edge_transition_examples(image, results_xlsx, **kwargs):
    save_png = kwargs.get('save_png', False)
    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    edge_detector = saved_kwargs.get("edge_detector", '')
    pixel_size = saved_kwargs.get("pixel_size", 0.0)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_transition_examples.png'))
    top_text = saved_kwargs.get("top_text", '')
    subset_size = saved_kwargs.get("subset_size", 0)
    section_length = saved_kwargs.get("section_length", 0)
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])

    int_results = pd.read_excel(results_xlsx, sheet_name='Transition analysis results')
    X = int_results['X']
    Y = int_results['Y']
    X_grads = int_results['X grad']
    Y_grads = int_results['Y grad']
    error_flags = int_results['error_flag']
    
    X_selected = X[error_flags==0]
    Y_selected = Y[error_flags==0]
    centers_selected = np.vstack((Y_selected, X_selected)).T
    X_grads_selected = X_grads[error_flags==0]
    Y_grads_selected = Y_grads[error_flags==0]
    center_grads_selected = np.vstack((Y_grads_selected, X_grads_selected)).T

    fig, axsf = plt.subplots(4,3, figsize=(20, 15))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.90, wspace=0.10, hspace=0.15)
    for (j,center), center_grad in zip(enumerate(centers_selected[0:3]), center_grads_selected[0:3]):
        res = estimate_edge_transition(image, center, center_grad,
                                  bounds = bounds,
                                  axs=axsf[0:2, j],
                                  subset_size = subset_size,
                                  section_length = section_length,
                                  disp_res=True)
    for (j,center), center_grad in zip(enumerate(centers_selected[-3:]), center_grads_selected[-3:]):
        res = estimate_edge_transition(image, center, center_grad,
                                  bounds = bounds,
                                  axs=axsf[2:4, j],
                                  subset_size = subset_size,
                                  section_length = section_length,
                                  disp_res=True)

    fig.suptitle('Sample Transitions found using ' + edge_detector + ' Edge Detector')
    if save_png:
        axsf[0,0].text(0.05, 1.05, save_fname, transform=axsf[0, 0].transAxes)
        fig.savefig(save_fname, dpi=300)


