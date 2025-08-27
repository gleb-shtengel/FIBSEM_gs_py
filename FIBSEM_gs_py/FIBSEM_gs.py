import numpy as np
import pandas as pd
import os
import socket
import platform
from pathlib import Path
import time
import glob
import re
import gc
from copy import deepcopy
import cv2

import psutil
import inspect

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from IPython.core.pylabtools import figsize, getfigs
from PIL import Image as PILImage
from PIL.TiffTags import TAGS

from struct import unpack
#from tqdm import tqdm_notebook as tqdm
from tqdm.notebook import tqdm

#import skimage
#print('skimage version: ', skimage.__version__)
#from skimage.measure import ransac  # this is temporarily patched by my own corrected copies

from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, warp
try:
    import skimage.external.tifffile as tiff
except:
    import tifffile as tiff
#from scipy import __version__ as scipy_version
#print('scipy version:   ', scipy_version)
from scipy.signal import savgol_filter
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

#from sklearn import __version__ as sklearn_version
#print('sklearn version: ', sklearn_version)
from sklearn.linear_model import (LinearRegression,
    TheilSenRegressor,
    RANSACRegressor,
    HuberRegressor,
    RidgeCV,
    LassoCV)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import dask
import dask.array as da
from dask.distributed import Client, progress, get_task_stream
from dask.diagnostics import ProgressBar
from dask.distributed import as_completed

#import cv2
#print('Open CV version: ', cv2. __version__)
import mrcfile
import h5py
import npy2bdv
import pickle
import webbrowser
from IPython.display import IFrame

EPS = np.finfo(float).eps

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
try:
    warnings.filterwarnings('ignore', category=np.RankWarning)
except:
    warnings.filterwarnings('ignore', category=np.exceptions.RankWarning)

try:
    from FIBSEM_gs_py.FIBSEM_help_functions_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_help_functions_gs")

try:
    from FIBSEM_gs_py.FIBSEM_custom_transforms_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_custom_transforms_gs")

try:
    from FIBSEM_gs_py.FIBSEM_resolution_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_resolution_gs")

try:
    from FIBSEM_gs_py.FIBSEM_streak_analysis_gs import *
except:
    raise RuntimeError("Unable to load FIBSEM_streak_analysis_gs")

################################################
# The two functions below are a patch on skimage.measure.ransac
################################################
def _dynamic_max_trials_gs(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    The temp fix was done.

    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.

    Returns
    -------
    trials : int
        Number of trials.
    """
    #if probability == 0:
    #    return 0
    if probability == 1:     # this is a temp fix
        return np.inf        # this is a temp fix
    if n_inliers == 0:
        return np.inf
    inlier_ratio = n_inliers / n_samples
    nom = max(_EPSILON, 1 - probability)
    denom = max(_EPSILON, 1 - inlier_ratio ** min_samples)
    return np.ceil(np.log(nom) / np.log(denom))

def ransac(
    data,
    model_class,
    min_samples,
    residual_threshold,
    is_data_valid=None,
    is_model_valid=None,
    max_trials=100,
    stop_sample_num=np.inf,
    stop_residuals_sum=0,
    stop_probability=1,
    rng=None,
    initial_inliers=None,
):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples value.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation

    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N,) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    >>> t = np.linspace(0, 2 * np.pi, 50)
    >>> xc, yc = 20, 30
    >>> a, b = 5, 10
    >>> x = xc + a * np.cos(t)
    >>> y = yc + b * np.sin(t)
    >>> data = np.column_stack([x, y])
    >>> rng = np.random.default_rng(203560)  # do not copy this value
    >>> data += rng.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    True
    >>> np.round(model.params)  # doctest: +SKIP
    array([ 72.,  75.,  77.,  14.,   1.])

    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    >>> abs(np.round(ransac_model.params))  # doctest: +SKIP
    array([20., 30., 10.,  6.,  2.])
    >>> inliers  # doctest: +SKIP
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)
    >>> sum(inliers) > 40
    True

    RANSAC can be used to robustly estimate a geometric
    transformation. In this section, we also show how to use a
    proportion of the total samples, rather than an absolute number.

    >>> from skimage.transform import SimilarityTransform
    >>> rng = np.random.default_rng()
    >>> src = 100 * rng.random((50, 2))
    >>> model0 = SimilarityTransform(scale=0.5, rotation=1,
    ...                              translation=(10, 20))
    >>> dst = model0(src)
    >>> dst[0] = (10000, 10000)
    >>> dst[1] = (-100, 100)
    >>> dst[2] = (50, 50)
    >>> ratio = 0.5  # use half of the samples
    >>> min_samples = int(ratio * len(src))
    >>> model, inliers = ransac(
    ...     (src, dst),
    ...     SimilarityTransform,
    ...     min_samples,
    ...     10,
    ...     initial_inliers=np.ones(len(src), dtype=bool),
    ... )  # doctest: +SKIP
    >>> inliers  # doctest: +SKIP
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True])

    """

    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = []
    validate_model = is_model_valid is not None
    validate_data = is_data_valid is not None

    rng = np.random.default_rng(rng)

    # in case data is not pair of input and output, male it like it
    if not isinstance(data, (tuple, list)):
        data = (data,)
    num_samples = len(data[0])

    if not (0 < min_samples <= num_samples):
        raise ValueError(f"`min_samples` must be in range (0, {num_samples}]")

    if residual_threshold < 0:
        raise ValueError("`residual_threshold` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if not (0 <= stop_probability <= 1):
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError(
            f"RANSAC received a vector of initial inliers (length "
            f"{len(initial_inliers)}) that didn't match the number of "
            f"samples ({num_samples}). The vector of initial inliers should "
            f"have the same length as the number of samples and contain only "
            f"True (this sample is an initial inlier) and False (this one "
            f"isn't) values."
        )

    # for the first run use initial guess of inliers
    spl_idxs = (
        initial_inliers
        if initial_inliers is not None
        else rng.choice(num_samples, min_samples, replace=False)
    )

    # estimate model for current random sample set
    model = model_class()

    num_trials = 0
    # max_trials can be updated inside the loop, so this cannot be a for-loop
    while num_trials < max_trials:
        num_trials += 1

        # do sample selection according data pairs
        samples = [d[spl_idxs] for d in data]

        # for next iteration choose random sample set and be sure that
        # no samples repeat
        spl_idxs = rng.choice(num_samples, min_samples, replace=False)

        # optional check if random sample set is valid
        if validate_data and not is_data_valid(*samples):
            continue

        success = model.estimate(*samples)
        # backwards compatibility
        if success is not None and not success:
            continue

        # optional check if estimated model is valid
        if validate_model and not is_model_valid(model, *samples):
            continue

        residuals = np.abs(model.residuals(*data))
        # consensus set / inliers
        inliers = residuals < residual_threshold
        residuals_sum = residuals.dot(residuals)

        # choose as new best model if number of inliers is maximal
        inliers_count = np.count_nonzero(inliers)
        if (
            # more inliers
            inliers_count > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (
                inliers_count == best_inlier_num
                and residuals_sum < best_inlier_residuals_sum
            )
        ):
            best_inlier_num = inliers_count
            best_inlier_residuals_sum = residuals_sum
            best_inliers = inliers
            max_trials = min(
                max_trials,
                _dynamic_max_trials_gs(
                    best_inlier_num, num_samples, min_samples, stop_probability
                ),
            )
            if (
                best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
            ):
                break

    # estimate final model using all inliers
    if any(best_inliers):
        # select inliers for each data array
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        if validate_model and not is_model_valid(model, *data_inliers):
            warn("Estimated model is not valid. Try increasing max_trials.")
    else:
        model = None
        best_inliers = None
        warn("No inliers found. Model not fitted")

    return model, best_inliers

def levinson_durbin(s, nlags=10, isacov=False):
    """
    Levinson-Durbin recursion for autoregressive processes.
    copied from:
    https://www.statsmodels.org/stable/_modules/statsmodels/tsa/stattools.html#levinson_durbin

    Parameters
    ----------
    s : array_like
        If isacov is False, then this is the time series. If iasacov is true
        then this is interpreted as autocovariance starting with lag 0.
    nlags : int, optional
        The largest lag to include in recursion or order of the autoregressive
        process.
    isacov : bool, optional
        Flag indicating whether the first argument, s, contains the
        autocovariances or the data series.

    Returns
    -------
    sigma_v : float
        The estimate of the error variance.
    arcoefs : ndarray
        The estimate of the autoregressive coefficients for a model including
        nlags.
    pacf : ndarray
        The partial autocorrelation function.
    sigma : ndarray
        The entire sigma array from intermediate result, last value is sigma_v.
    phi : ndarray
        The entire phi array from intermediate result, last column contains
        autoregressive coefficients for AR(nlags).

    Notes
    -----
    This function returns currently all results, but maybe we drop sigma and
    phi from the returns.

    If this function is called with the time series (isacov=False), then the
    sample autocovariance function is calculated with the default options
    (biased, no fft).
    """
    #s = array_like(s, "s")
    #nlags = int_like(nlags, "nlags")
    #isacov = bool_like(isacov, "isacov")

    order = nlags

    if isacov:
        sxx_m = s
    else:
        sxx_m = acovf(s, fft=False)[: order + 1]  # not tested

    phi = np.zeros((order + 1, order + 1), "d")
    sig = np.zeros(order + 1)
    # initial points for the recursion
    phi[1, 1] = sxx_m[1] / sxx_m[0]
    sig[1] = sxx_m[0] - phi[1, 1] * sxx_m[1]
    for k in range(2, order + 1):
        phi[k, k] = (
            sxx_m[k] - np.dot(phi[1:k, k - 1], sxx_m[1:k][::-1])
        ) / sig[k - 1]
        for j in range(1, k):
            phi[j, k] = phi[j, k - 1] - phi[k, k] * phi[k - j, k - 1]
        sig[k] = sig[k - 1] * (1 - phi[k, k] ** 2)

    sigma_v = sig[-1]
    arcoefs = phi[1:, -1]
    pacf_ = np.diag(phi).copy()
    pacf_[0] = 1.0
    return sigma_v, arcoefs, pacf_, sig, phi  # return everything


def gauss_with_offset(x, a, x0, b, sigma):
    return (a*np.exp(-(x-x0)**2/(2*sigma**2)) + b)


def gauss_without_offset(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


################################################
#      Single Frame Image Processing Functions
################################################

def find_autocorrelation_peak(ind_acr, mag_acr, **kwargs):
    '''
    Estimates the noise-free value of auto-correlation function by extrapolation.  ©G.Shtengel 12/2024 gleb.shtengel@gmail.com
    
    Parameters
    ---------
    ind_acr : 1D - array
        indices (coordinates in pixels)
    mag_acr : 1D - array
        values of Auto-Correlation function (suppozed to be zero-centered)

    kwargs:
    extrapolate_signal : str
        extrapolate to find signal autocorrelationb to 0-point (without noise). 
        Options are:
            'nearest'  - nearest point (1 pixel away from center, same as NN in [1]).
            'linear'   - linear interpolation of 2-points next to center (same as FO in [1]).
            'parabolic' - parabolic interpolation of 2 point left and 2 points right (for 4-point interpolation this is the same as NN+FO in [1]).
            'gaussian'  - gaussian interpolation with number of points = aperture
            'LDR' - use Levinson-Durbin recusrsion (ACLDR in [1]).
        Default is 'parabolic'.
    nlags : int
        in case of 'LDR' (Levinson-Durbin recusrsion) nlags is the recursion order (a number of lags)
    aperture : int
        total number of points for gaussian interpolation
    edge_fraction : float
        Fraction of the full auto-correlation  range used to calculate the "zero value" (default is 0.10).

    [1]. K. s. Sim, M. s. Lim, Z. x. Yeap, Performance of signal-to-noise ratio estimation for scanning electron microscope using autocorrelation Levinson–Durbin recursion model. J. Microsc. 263, 64–77 (2016).

    Returns: mag_acr_peak, mag_NFacr, ind_acr, mag_acr
    '''
    extrapolate_signal = kwargs.get('extrapolate_signal', 'parabolic')
    edge_fraction = kwargs.get("edge_fraction", 0.10)
    aperture = kwargs.get("aperture", 6)
    sz = len(ind_acr)
    nlags = kwargs.get("nlags", sz//4)
    
    ind_acr_l = ind_acr[sz//2-2:sz//2]
    ind_acr_c = ind_acr[sz//2]
    ind_acr_r = ind_acr[sz//2+1 : sz//2+3]
    mag_acr_left = mag_acr[(sz//2-2):(sz//2)]
    mag_acr_right = mag_acr[(sz//2+1):(sz//2+3)]
    
    if extrapolate_signal == 'LDR':
        half_ACR = mag_acr[sz//2:]
        sigma_v, ar_coefs, pacf, sigma , phi = levinson_durbin(half_ACR, nlags=nlags, isacov=True)
        mag_NFacr = np.sum(ar_coefs[0:nlags]*half_ACR[0:nlags])
    elif extrapolate_signal == 'gaussian':
        di = aperture//2
        ACR_nozero = np.concatenate((mag_acr[sz//2-di : sz//2], mag_acr[sz//2+1 : sz//2+di+1]))
        lags_nozero = np.concatenate((ind_acr[sz//2-di : sz//2], ind_acr[sz//2+1 : sz//2+di+1]))
        amp_guess = np.max(ACR_nozero)-np.min(ACR_nozero)
        center_guess = np.mean(lags_nozero)
        sigma_guess = find_FWHM(lags_nozero, ACR_nozero)[0] / 2.4
        offset_guess = 2.0 * np.min(ACR_nozero) - np.max(ACR_nozero)
        popt, pcov = curve_fit(gauss_with_offset, lags_nozero, ACR_nozero, p0=[amp_guess, center_guess, offset_guess, sigma_guess])
        mag_NFacr = popt[0]+popt[2]
    else:
        if extrapolate_signal == 'parabolic':
            mag_NFacr_l = (4 * mag_acr_left[1] - mag_acr_left[0]) / 3.0
            mag_NFacr_r = (4 * mag_acr_right[0] - mag_acr_right[1]) / 3.0
        elif extrapolate_signal == 'linear':
            mag_NFacr_l = 2 * mag_acr_left[1] - mag_acr_left[0]
            mag_NFacr_r = 2 * mag_acr_right[0] - mag_acr_right[1]
        else:
            mag_NFacr_l = mag_acr_left[1]
            mag_NFacr_r = mag_acr_right[0]
        mag_NFacr = (mag_NFacr_l + mag_NFacr_r)/2.0

    mag_acr_peak = mag_acr[sz//2]
    ind_acr = ind_acr[sz//2-2:sz//2+3]
    mag_acr = np.concatenate((mag_acr_left, np.array([mag_NFacr]), mag_acr_right))

    if extrapolate_signal == 'parabolic':
        coeff = np.polyfit(ind_acr, mag_acr, 2)
        ind_acr = np.linspace(ind_acr[0], ind_acr[-1], 50)
        mag_acr = np.polyval(coeff, ind_acr)

    if extrapolate_signal == 'gaussian':
        ind_acr = np.linspace(lags_nozero[0], lags_nozero[-1], 50)
        mag_acr = gauss_with_offset(ind_acr,*popt)

    return mag_acr_peak, mag_NFacr, ind_acr, mag_acr


def Single_Image_SNR(img, **kwargs):
    '''
    Estimates SNR based on a single image.
    ©G.Shtengel 12/2024 gleb.shtengel@gmail.com
    Calculates SNR of a single image based on auto-correlation analysis after [1].
    
    Parameters
    ---------
    img : 2D array
     
    kwargs:
    ----------
    edge_fraction : float
        Fraction of the full auto-correlation  range used to calculate the "zero value" (default is 0.10).
    extrapolate_signal : str
        Extrapolation method to find signal auto-correlation at 0-point (“Noise-Free Autocorrelation”). 
        Options are (default is 'parabolic'):
            'nearest'  - nearest point (1 pixel away from center)
            'linear'   - linear interpolation of 2-points next to center
            'parabolic' - parabolic interpolation of 2 points left and 2 points right 
            'gaussian'  - gaussian interpolation with number of points = aperture
            'LDR' - use Levinson-Durbin recusrsion (ACLDR in [1]).
    nlags : int
        In case of 'LDR' (Levinson-Durbin recursion) nlags is the recursion order (a number of lags). Default is min(xsize/4, ysize/4).
    aperture : int
        Total number of points for gaussian interpolation. Default is 6.
    zero_mean: boolean
        If True (default), auto-correlation is zero-mean.
    disp_res : boolean
        Display results (plots) (default is True).
    save_res_png : boolean
        Save the analysis output into a PNG file (default is True).
    res_fname : string
        filename for the result image. Default is 'SNR_result.png'.
    img_label : string
        Optional image label.
    dpi : int
        Resolution (DPI) for the output PNG image.
    verbose : boolean
        Display intermediate results.
        
    Returns:
        xSNR, ySNR, rSNR : float, float, float
            SNR determined using the method in [1] along X- and Y- directions.
            If there is a direction with slow varying data - that direction provides more accurate SNR estimate
            Y-streaks in typical FIB-SEM data provide slow varying Y-component because streaks
            usually get increasingly worse with increasing Y. 
            So for typical FIB-SEM data use ySNR
    
    [1] J. T. L. Thong et al, Single-image signal-to-noise ratio estimation. Scanning, 328–336 (2001).
    '''
    edge_fraction = kwargs.get("edge_fraction", 0.10)
    extrapolate_signal = kwargs.get('extrapolate_signal', 'parabolic')
    aperture = kwargs.get("aperture", 6)
    zero_mean = kwargs.get('zero_mean', True)
    disp_res = kwargs.get("disp_res", True)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", 'SNR_results.png')
    img_label = kwargs.get("img_label", 'Orig. Image')
    dpi = kwargs.get("dpi", 300)
    verbose = kwargs.get("verbose", False)
    
    #first make image size even
    ysz, xsz = img.shape
    img = np.float64(img[0:((ysz+1)//2*2-1), 0:((xsz+1)//2*2-1)])
    ysz, xsz = img.shape
    nlags = kwargs.get("nlags", np.min((ysz, xsz))//4)

    xy_ratio = xsz/ysz
    if zero_mean:
        data_FT = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img-img.mean())))
    else:
        data_FT = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
    data_FC = (np.multiply(data_FT,np.conj(data_FT)))/xsz/ysz
    data_ACR = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data_FC))))
    data_ACR_peak = data_ACR[ysz//2, xsz//2]
    data_ACR_log = np.log(data_ACR)
    data_ACR = data_ACR / data_ACR_peak
    radial_ACR = radial_profile(data_ACR, [xsz//2, ysz//2])
    r_ACR = np.concatenate((radial_ACR[::-1], radial_ACR[1:]))
    
    rsz = len(r_ACR)
    rcr = np.linspace(-rsz//2+1, rsz//2, rsz)
    xcr = np.linspace(-xsz//2+1, xsz//2, xsz)
    ycr = np.linspace(-ysz//2+1, ysz//2, ysz)

    if verbose:
        print('Extracting Noise-Free Autocorrelation value using ', extrapolate_signal)
    
    mag_acr_peak_x, mag_NFacr_x, ind_acr_x, mag_acr_x = find_autocorrelation_peak(xcr, data_ACR[ysz//2, :],
                                                                    extrapolate_signal = extrapolate_signal,
                                                                    aperture = aperture,
                                                                    nlags = nlags)
    mag_acr_peak_y, mag_NFacr_y, ind_acr_y, mag_acr_y = find_autocorrelation_peak(ycr, data_ACR[:, xsz//2],
                                                                    extrapolate_signal = extrapolate_signal,
                                                                    aperture = aperture,
                                                                    nlags = nlags)
    mag_acr_peak_r, mag_NFacr_r, ind_acr_r, mag_acr_r = find_autocorrelation_peak(rcr, r_ACR,
                                                                    extrapolate_signal = extrapolate_signal,
                                                                    aperture = aperture,
                                                                    nlags = nlags)
    
    xedge = np.int32(xsz*edge_fraction)
    yedge = np.int32(ysz*edge_fraction)
    redge = np.int32(rsz*edge_fraction)
    
    if zero_mean:
        mag_acr_mean_x = np.mean(data_ACR[ysz//2, 0:xedge])
        mag_acr_mean_y = np.mean(data_ACR[0:yedge, xsz//2])
        mag_acr_mean_r = np.mean(r_ACR[0:redge])
    else:    
        mag_acr_mean_x = 0.0
        mag_acr_mean_y = 0.0
        mag_acr_mean_r = 0.0
    ind_mag_acr_mean_x = np.linspace(-xsz//2, (-xsz//2+xedge-1), xedge)
    ind_mag_acr_mean_y = np.linspace(-ysz//2, (-ysz//2+yedge-1), yedge)
    ind_mag_acr_mean_r = np.linspace(-rsz//2, (-rsz//2+redge-1), redge)
    
    xSNR = (mag_NFacr_x - mag_acr_mean_x)/(mag_acr_peak_x - mag_NFacr_x)
    ySNR = (mag_NFacr_y - mag_acr_mean_y)/(mag_acr_peak_y - mag_NFacr_y)
    rSNR = (mag_NFacr_r - mag_acr_mean_r)/(mag_acr_peak_r - mag_NFacr_r)
    if disp_res:
        fs=12
        
        if xy_ratio < 2.5:
            fig, axs = plt.subplots(1,4, figsize = (20, 5))
        else:
            fig = plt.figure(figsize = (20, 5))
            ax0 = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 3)
            ax2 = fig.add_subplot(1, 4, 3)
            ax3 = fig.add_subplot(1, 4, 4)
            axs = [ax0, ax1, ax2, ax3]
        fig.subplots_adjust(left=0.03, bottom=0.06, right=0.99, top=0.92, wspace=0.25, hspace=0.10)
        
        range_disp = get_min_max_thresholds(img, thr_min = thresholds_disp[0], thr_max = thresholds_disp[1], nbins = nbins_disp, disp_res=False)
        axs[0].imshow(img, cmap='Greys', vmin=range_disp[0], vmax=range_disp[1])
        axs[0].grid(True)
        axs[0].set_title(img_label)
        if save_res_png:
            axs[0].text(0, 1.1 + (xy_ratio-1.0)/20.0, res_fname, transform=axs[0].transAxes)
        axs[1].imshow(data_ACR_log, extent=[-xsz//2-1, xsz//2, -ysz//2-1, ysz//2])
        axs[1].grid(True)
        axs[1].set_title('Autocorrelation (log scale)')
    
        axs[2].plot(xcr, data_ACR[ysz//2, :], 'r', linewidth=0.5, label='X')
        axs[2].plot(ycr, data_ACR[:, xsz//2], 'b', linewidth=0.5, label='Y')
        axs[2].plot(rcr, r_ACR, 'g', linewidth=0.5, label='R')
        axs[2].plot(ind_mag_acr_mean_x, ind_mag_acr_mean_x*0 + mag_acr_mean_x, 'r--', linewidth =2.0, label='<X>={:.5f}'.format(mag_acr_mean_x))
        axs[2].plot(ind_mag_acr_mean_y, ind_mag_acr_mean_y*0 + mag_acr_mean_y, 'b--', linewidth =2.0, label='<Y>={:.5f}'.format(mag_acr_mean_y))
        axs[2].plot(ind_mag_acr_mean_r, ind_mag_acr_mean_r*0 + mag_acr_mean_r, 'g--', linewidth =2.0, label='<R>={:.5f}'.format(mag_acr_mean_r))
        axs[2].grid(True)
        axs[2].legend()
        axs[2].set_title('Normalized autocorr. cross-sections')
        axs[3].plot(xcr, data_ACR[ysz//2, :], 'rx', label='X data')
        axs[3].plot(ycr, data_ACR[:, xsz//2], 'bd', label='Y data')
        axs[3].plot(rcr, r_ACR, 'g+', ms=10, label='R data')
        axs[3].plot(xcr[xsz//2], data_ACR[ysz//2, xsz//2], 'md', label='Peak: {:.4f}, {:.4f}'.format(xcr[xsz//2], data_ACR[ysz//2, xsz//2]))
        axs[3].plot(ind_acr_x, mag_acr_x, 'r', label='X extrap.: {:.4f}, {:.4f}'.format(ind_acr_x[len(ind_acr_x)//2], mag_acr_x[len(mag_acr_x)//2]))
        axs[3].plot(ind_acr_y, mag_acr_y, 'b', label='Y extrap.: {:.4f}, {:.4f}'.format(ind_acr_y[len(ind_acr_y)//2], mag_acr_y[len(mag_acr_y)//2]))
        axs[3].plot(ind_acr_r, mag_acr_r, 'g', label='R extrap.: {:.4f}, {:.4f}'.format(ind_acr_r[len(ind_acr_r)//2], mag_acr_r[len(mag_acr_r)//2]))
        axs[3].text(0.03, 0.96, 'Noise-free peak extr.: '+extrapolate_signal, transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.03, 0.90, 'xSNR = {:.2f}'.format(xSNR), color='r', transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.03, 0.84, 'ySNR = {:.2f}'.format(ySNR), color='b', transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.03, 0.78, 'rSNR = {:.2f}'.format(rSNR), color='g', transform=axs[3].transAxes, fontsize=fs)
        axs[3].grid(True)
        axs[3].legend()
        axs[3].set_xlim(-5,5)
        y_ar = np.concatenate((mag_acr_x, mag_acr_y, mag_acr_r, np.array((mag_acr_peak_x, mag_acr_peak_y, mag_acr_peak_r))))
        yi = np.min(y_ar) - 0.5*(np.max(y_ar) - np.min(y_ar))
        ya = np.max(y_ar) + 0.2*(np.max(y_ar) - np.min(y_ar))
        axs[3].set_ylim((yi, ya))
        axs[3].set_title('Normalized autocorr. cross-sections')

        if save_res_png:
            #print('X:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(mag_acr_peak_x, mag_NFacr_x, mag_acr_mean_x ))
            #print('xSNR = {:.2f}'.format(xSNR))
            #print('Y:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(y_acr, y_noise_free_acr, mag_acr_mean_y ))
            #print('ySNR = {:.4f}'.format(ySNR))
            #print('R:   ACR peak = {:.4f}, Noise-Free ACR Peak = {:.4f}, Squared Mean = {:.4f}'.format(r_acr, r_noise_free_acr, mag_acr_mean_r ))
            #print('rSNR = {:.4f}'.format(rSNR))
            fig.savefig(res_fname, dpi=dpi)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saved the results into the file: ', res_fname)
    
    return xSNR, ySNR, rSNR


def Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):
    '''
    Analyses the noise statistics in the selected ROI's of the EM data
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    This function is somewhat obsolete. When possible – use Single_Image_SNR(img, **kwargs) , it is more powerful and accurate way of analysing the noise distribution.
    Performs following:
    1. For each of the selected Noise_ROIs, this method will perform the following:
        1a. Smooth the data by 2D convolution with a given kernel.
        1b. Determine "Noise" as difference between the original raw and smoothed data.        
        1c. Calculate the mean intensity value of the data and variance of the above "Noise"
    2. Plot the dependence of the noise variance vs. image intensity.
    3. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
        it will be determined from the header data:
            for RawImageA it is self.Scaling[1,0]
            for RawImageB it is self.Scaling[1,1]
    4. The linear equation is determined for a line that passes through the point:
            Intensity=DarkCount and Noise Variance = 0
            and is the best fit for the [Mean Intensity, Noise Variance] points
            determined for each ROI (Step 1 above).
    5. Another ROI (defined by Hist_ROI parameter) is used to build an
        intensity histogram of the actual data. Peak of that histogram is determined.
    6. The data is plotted. Two values of SNR are defined from the slope of the line in Step 4:
        PSNR (Peak SNR) = Mean Intensity/sqrt(Noise Variance) at the intensity
            at the histogram peak determined in the Step 5.
        DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
            where Max and Min Intensity are determined by corresponding cumulative
            threshold parameters, and Noise Variance is taken at the intensity
            in the middle of the range (Min Intensity + Max Intensity)/2.0

    Parameters
    ----------
    img : 2D array
        Original image
    Noise_ROIs : list of lists: [[left, right, top, bottom]]
        List of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the noise.
    Hist_ROI : list [left, right, top, bottom]
        Coordinates (indices) of the boundaries of the image subset to evaluate the real data histogram.

    kwargs:
    ----------
    DarkCount : float
        The value of the Intensity Data at 0.
    kernel : 2D float array
        A kernel to perform 2D smoothing convolution. Default is normalized np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]), where st = 1/np.sqrt(2).
    nbins_disp : int
        Number of histogram bins for building the PDF and CDF to determine the data range for data display. Default is 256.
    thresholds_disp : list [thr_min_disp, thr_max_disp]
        CDF threshold for determining the min and max data values for display. Default id [1e-3, 1e-3].
    nbins_analysis : int
        Number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5. Default is 256.
    thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
        CDF threshold for building the data histogram in Step 5. Default is [2e-2, 2e-2].
    disp_res : boolean
        If True - plot/ display the results. Default is True.
    save_res_png : boolean
        Save the analysis output into a PNG file. Default is True.
    res_fname : string
        Filename - used for plotting the data. Default is 'SNR_result.png'.
    img_label : string
        Optional image label.
    Notes : string
        Optional additional notes.
    dpi : int
        Resolution (DPI) of the PNG image.
        
    Returns:
    ----------
    mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR
        mean_vals and var_vals are the Mean Intensity and Noise Variance values for the Noise_ROIs (Step 1)
        NF_slope is the slope of the linear fit curve (Step 4)
        PSNR and DSNR are Peak and Dynamic SNR's (Step 6)
    '''
    st = 1.0/np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
    def_kernel = def_kernel/def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    DarkCount = kwargs.get("DarkCount", 0)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    nbins_analysis = kwargs.get("nbins_analysis", 100)
    thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
    disp_res = kwargs.get("disp_res", True)
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", 'Noise_Analysis_ROIs.png')
    img_label = kwargs.get("img_label", '')
    Notes = kwargs.get("Notes", '')
    dpi = kwargs.get("dpi", 300)

    fs=10
    img_filtered = convolve2d(img, kernel, mode='same')
    range_disp = get_min_max_thresholds(img_filtered, thr_min = thresholds_disp[0], thr_max = thresholds_disp[1], nbins = nbins_disp, disp_res=False)

    xi, xa, yi, ya = Hist_ROI
    img_hist = img[yi:ya, xi:xa]
    img_hist_filtered = img_filtered[yi:ya, xi:xa]
    
    range_analysis = get_min_max_thresholds(img_hist_filtered, thr_min = thresholds_analysis[0], thr_max = thresholds_analysis[1], nbins = nbins_analysis, disp_res=False)
    if disp_res:
        print('The EM data range for noise analysis: {:.1f} - {:.1f},  DarkCount={:.1f}'.format(range_analysis[0], range_analysis[1], DarkCount))
    bins_analysis = np.linspace(range_analysis[0], range_analysis[1], nbins_analysis)
    
    yx_ratio = img.shape[0]/img.shape[1]
    xsz = 15.0
    ysz = xsz*2.0*yx_ratio + 5
    xsz = xsz / max((xsz, ysz)) * 15.0
    ysz = ysz / max((xsz, ysz)) * 15.0  
    widths = [1.0, 1.0, 1.0]
    heights = [xsz/5.0*yx_ratio, xsz/5.0*yx_ratio, 0.7]

    n_ROIs = len(Noise_ROIs)+1
    mean_vals = np.zeros(n_ROIs)
    var_vals = np.zeros(n_ROIs)
    mean_vals[0] = DarkCount

    if disp_res: 
        fig = plt.figure(figsize=(xsz,ysz))
        gr_spec = fig.add_gridspec(ncols=3, nrows=3, width_ratios=widths, height_ratios=heights)
        axs0 = fig.add_subplot(gr_spec[0, :])
        axs1 = fig.add_subplot(gr_spec[1, :])
        axs2 = fig.add_subplot(gr_spec[2, 0])
        axs3 = fig.add_subplot(gr_spec[2, 1])
        axs4 = fig.add_subplot(gr_spec[2, 2])
        fig.subplots_adjust(left=0.01, bottom=0.06, right=0.99, top=0.95, wspace=0.25, hspace=0.10)
       
        axs0.text(0.01, 1.13, res_fname + ',   ' +  Notes, transform=axs0.transAxes, fontsize=fs-3)
        axs0.imshow(img, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs0.axis(False)
        axs0.set_title('Original Image: ' + img_label, color='r', fontsize=fs)
        Hist_patch = patches.Rectangle((xi,yi), np.abs(xa-xi)-2, np.abs(ya-yi)-2, linewidth=1.0, edgecolor='white',facecolor='none')
        axs1.add_patch(Hist_patch)
        
        axs2.imshow(img_hist_filtered, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs2.axis(False)
        axs2.set_title('Smoothed ROI', fontsize=fs)
        
    if disp_res:
        hist, bins, hist_patches = axs3.hist(img_hist_filtered.ravel(), range=range_disp, bins = nbins_disp)
    else:
        hist, bins = np.histogram(img_hist_filtered.ravel(), range=range_disp, bins = nbins_disp)

    bin_centers = np.array(bins[1:] - (bins[1]-bins[0])/2.0)
    hist_center_ind = np.argwhere((bin_centers>range_analysis[0]) & (bin_centers<range_analysis[1]))
    hist_smooth = savgol_filter(np.array(hist), (nbins_disp//10)*2+1, 7)
    I_peak = bin_centers[hist_smooth.argmax()]
    I_mean = np.mean(img)
    C_peak = hist_smooth.max()

    if disp_res:
        axs3.plot(bin_centers[hist_center_ind], hist_smooth[hist_center_ind], color='grey', linestyle='dashed', linewidth=2)
        Ipeak_lbl = '$I_{peak}$' +'={:.1f}'.format(I_peak)
        axs3.plot(I_peak, C_peak, 'rd', label = Ipeak_lbl)
        axs3.set_title('Histogram of the Smoothed ROI', fontsize=fs)
        axs3.grid(True)
        axs3.set_xlabel('Smoothed ROI Image Intensity', fontsize=fs)
        for hist_patch in np.array(hist_patches)[bin_centers<range_analysis[0]]:
            hist_patch.set_facecolor('lime')
        for hist_patch in np.array(hist_patches)[bin_centers>range_analysis[1]]:
            hist_patch.set_facecolor('red')
        ylim3=np.array(axs3.get_ylim())
        I_min, I_max = range_analysis 
        axs3.plot([I_min, I_min],[ylim3[0]-1000, ylim3[1]], color='lime', linestyle='dashed', label='$I_{min}$' +'={:.1f}'.format(I_min))
        axs3.plot([I_max, I_max],[ylim3[0]-1000, ylim3[1]], color='red', linestyle='dashed', label='$I_{max}$' +'={:.1f}'.format(I_max))
        axs3.set_ylim(ylim3)
        axs3.legend(loc='upper right', fontsize=fs)
        axs1.imshow(img_filtered, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs1.axis(False)
        axs1.set_title('Smoothed Image')
        axs4.plot(DarkCount, 0, 'd', color = 'black', label='Dark Count')
    
    for j, ROI in enumerate(tqdm(Noise_ROIs, desc = 'analyzing ROIs')):
        xi, xa, yi, ya = ROI
        img_ROI = img[yi:ya, xi:xa]
        img_ROI_filtered = img_filtered[yi:ya, xi:xa]

        imdiff = (img_ROI - img_ROI_filtered)
        x = np.mean(img_ROI)
        y = np.var(imdiff)
        mean_vals[j+1] = x
        var_vals[j+1] = y
        
        if disp_res:
            patch_col = plt.get_cmap("gist_rainbow_r")((j)/(n_ROIs))
            rect_patch = patches.Rectangle((xi,yi), np.abs(xa-xi)-2, np.abs(ya-yi)-2, linewidth=0.5, edgecolor=patch_col,facecolor='none')
            axs0.add_patch(rect_patch)
            axs4.plot(x, y, 'd', color = patch_col) #, label='patch {:d}'.format(j))
    
    NF_slope = np.mean(var_vals[1:]/(mean_vals[1:]-DarkCount))
    mean_val_fit = np.array([mean_vals.min(), mean_vals.max()])
    var_val_fit = (mean_val_fit-DarkCount)*NF_slope
    VAR = (I_peak - DarkCount) * NF_slope
    VAR_at_mean = ((I_max + I_min) /2.0 - DarkCount) * NF_slope
    PSNR = (I_peak - DarkCount)/np.sqrt(VAR)
    MSNR = (I_mean - DarkCount)/np.sqrt((I_mean - DarkCount)*NF_slope)
    DSNR = (I_max - I_min)/np.sqrt(VAR_at_mean)
    
    if disp_res:
        axs4.grid(True)
        axs4.set_title('Noise Distribution', fontsize=fs)
        axs4.set_xlabel('ROI Image Intensity Mean', fontsize=fs)
        axs4.set_ylabel('ROI Image Intensity Variance', fontsize=fs)
        axs4.plot(mean_val_fit, var_val_fit, color='orange', label='Fit:  y = (x {:.1f}) * {:.2f}'.format(DarkCount, NF_slope))
        axs4.legend(loc = 'upper left', fontsize=fs)
        ylim4=np.array(axs4.get_ylim())
        V_min = (I_min-DarkCount)*NF_slope
        V_max = (I_max-DarkCount)*NF_slope
        V_peak = (I_peak-DarkCount)*NF_slope
        axs4.plot([I_min, I_min],[ylim4[0], V_min], color='lime', linestyle='dashed', label='$I_{min}$' +'={:.1f}'.format(I_min))
        axs4.plot([I_max, I_max],[ylim4[0], V_max], color='red', linestyle='dashed', label='$I_{max}$' +'={:.1f}'.format(I_max))
        axs4.plot([I_peak, I_peak],[ylim4[0], V_peak], color='black', linestyle='dashed', label='$I_{peak}$' +'={:.1f}'.format(I_peak))
        axs4.set_ylim(ylim4)
        txt1 = 'Peak Intensity:  {:.1f}'.format(I_peak)
        axs4.text(0.05, 0.65, txt1, transform=axs4.transAxes, fontsize=fs)
        txt2 = 'Variance={:.1f}, STD={:.1f}'.format(VAR, np.sqrt(VAR))
        axs4.text(0.05, 0.55, txt2, transform=axs4.transAxes, fontsize=fs)
        txt3 = 'PSNR = {:.2f}'.format(PSNR)
        axs4.text(0.05, 0.45, txt3, transform=axs4.transAxes, fontsize=fs)
        txt3 = 'DSNR = {:.2f}'.format(DSNR)
        axs4.text(0.05, 0.35, txt3, transform=axs4.transAxes, fontsize=fs)
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   results saved into the file: '+res_fname)

    return mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR


def Single_Image_Noise_Statistics(img, **kwargs):
    '''
    Analyses the noise statistics of the EM data image.
    ©G.Shtengel 10/2024 gleb.shtengel@gmail.com
    
    Performs following:
    1. Smooth the image by 2D convolution with a given kernel.
    2. Determine "Noise Image" as difference between the original raw and smoothed data.
    3. Select subsets of original, smoothed and noise images by selecting only elements where the filter_array (optional input 2D array) is True. Use calculate_gradent_map(img, ** kwargs) to define filter_array.
    4. Build a histogram of Smoothed Image (subset if filter_array was set).
    5. For all pixels withing each histogram bin of the Smoothed Image from Step 4, calculate the mean value (mean of intensities of the select pixels of the Smoothed Image from Step 4) and variance (variance of intensities of the select pixels of the Noise Image from the Step 2).
    6. Plot the dependence of the image variance vs. image intensity.
    7. Perform free linear fit of the variance vs. intensity. SNR0 is calculated as <S^2>/<N^2>.
    8. Perform linear fit with forced zero Intercept (DarkCount) of the variance vs. intensity. SNR1 is calculated <S^2>/<N^2>.

    Parameters
    ----------
    img : 2d array

    kwargs:
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image noise statistics.
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    DarkCount : float
        The value of the Intensity Data at 0.
    filter_array : 2d boolean array
        Array of the same dimensions as img. Only the pixel with corresponding filter_array values of True will be considered in the noise analysis.
    kernel : 2D float array
        A kernel to perform 2D smoothing convolution. Default is normalized np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]), where st = 1/np.sqrt(2).
    nbins_disp : int
        Number of histogram bins for building the PDF and CDF to determine the data range for data display. Default is 256.
    thresholds_disp : list [thr_min_disp, thr_max_disp]
        CDF threshold for determining the min and max data values for display. Default id [1e-3, 1e-3].
    nbins_analysis : int
        Number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5. Default is 256.
    thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
        CDF threshold for building the data histogram in Step 5. Default is [2e-2, 2e-2].
    disp_res : boolean
        If True - plot/ display the results. Default is True.
    disp_res_SNR0 : boolean
        Add SNR0 (free fit) results to Variance vs. Intensity plot. Default is True.
    disp_res_SNR1 : boolean
        Add SNR1 (use Dark Count from Scaling data) results to Variance vs. Intensity plot. Default is True.
    save_res_png : boolean
        Save the analysis output into a PNG file. Default is True.
    res_fname : str
        Filename - used for plotting the data. Default is'Noise_Analysis.png'
    img_label : string
        Optional image label.
    Notes : string
        Optional additional notes.
    dpi : int
        Resolution (DPI) of the PNG image.

    Returns:
    mean_vals, var_vals, I0, SNR0, SNR1, popt, result
        mean_vals and var_vals are the Mean Intensity and Noise Variance values for Step5, I0 is zero intercept (should be close to DarkCount)
        SNR0 and SNR1 are SNR's (Step 7 and 8 respectively)
    '''
    st = 1.0/np.sqrt(2.0)
    def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    def_kernel = def_kernel/def_kernel.sum()
    kernel = kwargs.get("kernel", def_kernel)
    DarkCount = kwargs.get("DarkCount", 0)
    nbins_disp = kwargs.get("nbins_disp", 256)
    thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
    nbins_analysis = kwargs.get("nbins_analysis", 100)
    thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
    disp_res = kwargs.get("disp_res", True)
    disp_res_SNR0 = kwargs.get("disp_res_SNR0", True)
    disp_res_SNR1 = kwargs.get("disp_res_SNR1", True)
    save_res_png = kwargs.get("save_res_png", True)
    res_fname = kwargs.get("res_fname", 'Noise_Analysis.png')
    image_name = kwargs.get("image_name", '')
    Notes = kwargs.get("Notes", '')
    dpi = kwargs.get("dpi", 300)
    filter_array = kwargs.get('filter_array', (img*0+1)>0)

    xi = 0
    yi = 0
    ysz, xsz = img.shape
    xa = xi + xsz
    ya = yi + ysz

    xi_eval = xi + evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = xa
    yi_eval = yi + evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ya

    img = img[yi_eval:ya_eval, xi_eval:xa_eval]
    img_smoothed = convolve2d(img, kernel, mode='same')[1:-1, 1:-1]
    img = img[1:-1, 1:-1]
    filter_array = filter_array[yi_eval:ya_eval, xi_eval:xa_eval][1:-1, 1:-1]
    
    imdiff = (img-img_smoothed)
    img_smoothed_filtered = img_smoothed[filter_array]
    imdiff_filtered = imdiff[filter_array]
    
    range_disp = get_min_max_thresholds(img_smoothed, thr_min = thresholds_disp[0], thr_max = thresholds_disp[1], nbins = nbins_disp, disp_res = False)       
    #range_analysis0 = get_min_max_thresholds(img_smoothed, thr_min = thresholds_analysis[0], thr_max = thresholds_analysis[1], nbins = nbins_analysis, disp_res = False)
    range_analysis = get_min_max_thresholds(img_smoothed_filtered, thr_min = thresholds_analysis[0], thr_max = thresholds_analysis[1], nbins = nbins_analysis, disp_res = False)
    
    if disp_res:
        #print('Length of original image is: ', np.prod(img_smoothed.shape))
        #print('Length of filtered image is: ', np.prod(img_smoothed_filtered.shape))
        print('')
        print('The EM data range for display:            {:.2f} to {:.2f}'.format(range_disp[0], range_disp[1]))
        #print('The EM data range0 for noise analysis:    {:.2f} to {:.2f}'.format(range_analysis0[0], range_analysis0[1]))
        print('The EM data range for noise analysis:     {:.2f} to {:.2f}'.format(range_analysis[0], range_analysis[1]))
    
    bins_analysis = np.linspace(range_analysis[0], range_analysis[1], nbins_analysis)
    range_imdiff = get_min_max_thresholds(imdiff, thr_min = thresholds_disp[0], thr_max = thresholds_disp[1], nbins = nbins_disp, disp_res = False)
    ind_new = np.digitize(img_smoothed_filtered, bins_analysis)
    
    result = np.array([(np.mean(img_smoothed_filtered[ind_new == j]), np.var(imdiff_filtered[ind_new == j]))  for j in range(1, nbins_analysis)])
    non_nan_ind = np.argwhere(np.invert(np.isnan(result[:, 0])))
    mean_vals = np.squeeze(result[non_nan_ind, 0])
    var_vals = np.squeeze(result[non_nan_ind, 1])
    
    xsz=15.0
    yx_ratio = img.shape[0]/img.shape[1]
    ysz = xsz/3.0*yx_ratio + 5.0
    xsz = xsz / max((xsz, ysz)) * 15.0
    ysz = ysz / max((xsz, ysz)) * 15.0

    low_mask = img*0.0+255.0
    high_mask = low_mask.copy()
    filter_mask = low_mask.copy()
    low_mask[img_smoothed > range_analysis[0]] = np.nan
    high_mask[img_smoothed < range_analysis[1]] = np.nan
    filter_mask[filter_array==True] = np.nan
    
    # np.product is deprecated, use np.prod instead
    #filter_nonzero = np.product(imdiff_filtered.shape)<np.product(imdiff.shape)
    filter_nonzero = np.prod(imdiff_filtered.shape)<np.prod(imdiff.shape)

    if disp_res:
        fs=11
        fig, axss = plt.subplots(2,3, figsize=(xsz,ysz),  gridspec_kw={"height_ratios" : [yx_ratio, 1.0]})
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.99, top=0.90, wspace=0.15, hspace=0.10)
        axs = axss.ravel()
        axs[0].text(-0.1, (0.98 + 0.1/yx_ratio), res_fname + ',       ' +  Notes, transform=axs[0].transAxes, fontsize=fs-2)

        axs[0].imshow(img, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs[0].axis(False)
        if len(image_name)>1:
            axs[0].set_title('Original Image: ' + image_name, color='r', fontsize=fs+1)
        else:
            axs[0].set_title('Original Image' + image_name, color='r', fontsize=fs+1)

        axs[1].imshow(img_smoothed, cmap="Greys", vmin = range_disp[0], vmax = range_disp[1])
        axs[1].axis(False)
        axs[1].set_title('Smoothed Image')

        axs[1].imshow(low_mask, cmap="brg_r")
        axs[1].imshow(high_mask, cmap="gist_rainbow")

        axs[2].imshow(imdiff, cmap="Greys", vmin = range_imdiff[0], vmax = range_imdiff[1])
        axs[2].axis(False)
        
        if filter_nonzero:
            axs[2].imshow(filter_mask, cmap="gist_rainbow")
            axs[2].text(0.0, 1.01, 'Image Difference', transform=axs[2].transAxes, fontsize=fs+1)
            axs[2].text(0.4, 1.01, 'Excluded pixels masked red', transform=axs[2].transAxes, color='red', fontsize=fs+1)
        else:
            axs[2].set_title('Image Difference', fontsize=fs+1)

    if disp_res:
        hist, bins, patches = axs[4].hist(img_smoothed_filtered.ravel(), range=range_disp, bins = nbins_disp)
    else:
        hist, bins = np.histogram(img_smoothed_filtered.ravel(), range=range_disp, bins = nbins_disp)
    bin_centers = np.array(bins[1:] - (bins[1]-bins[0])/2.0)
    hist_center_ind = np.argwhere((bin_centers>range_analysis[0]) & (bin_centers<range_analysis[1]))
    hist_smooth = savgol_filter(np.array(hist), (nbins_disp//10)*2+1, 7)
    I_peak = bin_centers[hist_smooth.argmax()]
    C_peak = hist_smooth.max()
    Ipeak_lbl = '$I_{peak}$' +'={:.1f}'.format(I_peak)
    
    if disp_res:
        axs[4].plot(bin_centers[hist_center_ind], hist_smooth[hist_center_ind], color='grey', linestyle='dashed', linewidth=2)
        axs[4].plot(I_peak, C_peak, 'rd', label = Ipeak_lbl)
        axs[4].legend(loc='upper left', fontsize=fs+1)
        if filter_nonzero:
            axs[4].set_title('Histogram of the Filtered Smoothed Image', fontsize=fs+1)
        else:
            axs[4].set_title('Histogram of the Smoothed Image', fontsize=fs+1)
        axs[4].grid(True)
        axs[4].set_xlabel('Image Intensity', fontsize=fs+1)
        for patch in np.array(patches)[bin_centers<range_analysis[0]]:
            patch.set_facecolor('lime')
        for patch in np.array(patches)[bin_centers>range_analysis[1]]:
            patch.set_facecolor('red')
        ylim4=np.array(axs[4].get_ylim())
        axs[4].plot([range_analysis[0], range_analysis[0]],[ylim4[0]-1000, ylim4[1]], color='lime', linestyle='dashed', label='Ilow')
        axs[4].plot([range_analysis[1], range_analysis[1]],[ylim4[0]-1000, ylim4[1]], color='red', linestyle='dashed', label='Ihigh')
        axs[4].set_ylim(ylim4)
        txt1 = 'Smoothing Kernel'
        axs[4].text(0.69, 0.955, txt1, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-1)
        txt2 = '{:.3f}  {:.3f}  {:.3f}'.format(kernel[0,0], kernel[0,1], kernel[0,2])
        axs[4].text(0.69, 0.910, txt2, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-2)
        txt3 = '{:.3f}  {:.3f}  {:.3f}'.format(kernel[1,0], kernel[1,1], kernel[1,2])
        axs[4].text(0.69, 0.865, txt3, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-2)
        txt3 = '{:.3f}  {:.3f}  {:.3f}'.format(kernel[2,0], kernel[2,1], kernel[2,2])
        axs[4].text(0.69, 0.820, txt3, transform=axs[4].transAxes, backgroundcolor='white', fontsize=fs-2)
    
    if disp_res:
        hist, bins, patches = axs[5].hist(imdiff_filtered.ravel(), bins = nbins_disp)
        axs[5].grid(True)
        if filter_nonzero:
            axs[5].set_title('Histogram of the Filtered Difference Map', fontsize=fs+1)
        else:
            axs[5].set_title('Histogram of the Difference Map', fontsize=fs+1)
        axs[5].set_xlabel('Image Difference', fontsize=fs+1) 
    else:
        hist, bins = np.histogram(imdiff_filtered.ravel(), bins = nbins_disp)
        
    try:
        popt = np.polyfit(mean_vals, var_vals, 1)
        I_array = np.array((range_analysis[0], range_analysis[1], I_peak))
        Var_array = np.polyval(popt, I_array)
        Var_peak = Var_array[2]
    except:
        if disp_res:
            print("np.polyfit could not converge")
        popt = np.array([np.var(imdiff)/np.mean(img_smoothed-DarkCount), 0])
        I_array = np.array((range_analysis[0], range_analysis[1], I_peak))
        Var_peak = np.var(imdiff)
    var_fit = np.polyval(popt, mean_vals)
    I0 = -popt[1]/popt[0]
    Slope_header = np.mean(var_vals/(mean_vals-DarkCount))
          
    var_fit_header = (mean_vals-DarkCount) * Slope_header
    if disp_res:
        axs[3].plot(mean_vals, var_vals, 'r.', label='data')
        if disp_res_SNR0:
            axs[3].plot(mean_vals, var_fit, 'b', label='linear fit: {:.1f}*x + {:.1f}'.format(popt[0], popt[1]))
        ylim3=np.array(axs[3].get_ylim())
        if disp_res_SNR1:
            axs[3].plot(mean_vals, var_fit_header, 'magenta', label='linear fit: {:.1f}*x + {:.1f}'.format(Slope_header, -Slope_header*DarkCount))
        axs[3].grid(True)
        axs[3].set_title('Noise Distribution', fontsize=fs+1)
        axs[3].set_xlabel('Image Intensity Mean', fontsize=fs+1)
        axs[3].set_ylabel('Image Intensity Variance', fontsize=fs+1)
        lbl_low = '$I_{low}$'+', thr={:.1e}'.format(thresholds_analysis[0])
        lbl_high = '$I_{high}$'+', thr={:.1e}'.format(thresholds_analysis[1])
        axs[3].plot([range_analysis[0], range_analysis[0]],[ylim3[0]-1000, ylim3[1]], color='lime', linestyle='dashed', label=lbl_low)
        axs[3].plot([range_analysis[1], range_analysis[1]],[ylim3[0]-1000, ylim3[1]], color='red', linestyle='dashed', label=lbl_high)
        axs[3].legend(loc='upper center', fontsize=fs+1)
        axs[3].set_ylim(ylim3)
    
    PSNR = (I_peak-I0)/np.sqrt(Var_peak)
    PSNR_header = (I_peak-DarkCount)/np.sqrt(Var_peak)
    DSNR = (range_analysis[1]-range_analysis[0])/np.sqrt(Var_peak)
    
    img_smoothed_filtered_resc = (img_smoothed_filtered - I0)/popt[0]
    imdiff_filtered_resc = imdiff_filtered / popt[0]
    #SNR0 = np.mean(img_smoothed_filtered_resc*img_smoothed_filtered_resc)/np.mean(img_smoothed_filtered_resc)
    SNR0 = np.mean(img_smoothed_filtered_resc*img_smoothed_filtered_resc)/np.var(imdiff_filtered_resc)
    img_smoothed_filtered_resc1 = (img_smoothed_filtered - DarkCount)/Slope_header
    imdiff_filtered_resc1 = imdiff_filtered / Slope_header
    #SNR1 = np.mean(img_smoothed_filtered_resc1*img_smoothed_filtered_resc1)/np.mean(img_smoothed_filtered_resc1)
    SNR1 = np.mean(img_smoothed_filtered_resc1*img_smoothed_filtered_resc1)/np.var(imdiff_filtered_resc1)
    
    if disp_res:
        print('')
        print('Used Dark Count Offset: {:.2f}'.format(DarkCount))
        print('Slope of linear fit with header offset: {:.2f}'.format(Slope_header))
        print('Fit w DarkCount  : SNR1 <S^2>/<N^2> = {:.2f}'.format(SNR1))
        print('')
        print('Free Fit Offset: {:.2f}'.format(I0))
        print('Slope of Free Fit: {:.2f}'.format(popt[0]))
        print('Free Fit         : SNR0 <S^2>/<N^2> = {:.2f}'.format(SNR0))
        
        if disp_res_SNR0:
            txt1 = 'Zero Int, Free Fit:    ' +'$I_{0}$' +'={:.1f}'.format(I0)
            axs[3].text(0.35, 0.17, txt1, transform=axs[3].transAxes, color='blue', fontsize=fs+1)
            txt2 = 'SNR0 <$S^2$>/<$N^2$> = {:.2f}'.format(SNR0)
            axs[3].text(0.35, 0.12, txt2, transform=axs[3].transAxes, color='blue', fontsize=fs+1)
        
        if disp_res_SNR1:
            txt3 = 'Zero Int, Dark Cnt.:    ' +'$I_{0}$' +'={:.1f}'.format(DarkCount)
            axs[3].text(0.35, 0.07, txt3, transform=axs[3].transAxes, color='magenta', fontsize=fs+1)
            txt4 = 'SNR1 <$S^2$>/<$N^2$> = {:.2f}'.format(SNR1)
            axs[3].text(0.35, 0.02, txt4, transform=axs[3].transAxes, color='magenta', fontsize=fs+1)

        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   results saved into the file: '+res_fname)
    return mean_vals, var_vals, I0, SNR0, SNR1, popt, result


def Perform_2D_fit(img, estimator, **kwargs):
    '''
    Bin the image and then perform 2D polynomial fit on the binned image.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    
    Parameters:
    ----------
    img : 2D array
        Original image.
    estimator : RANSACRegressor(),
                LinearRegression(),
                TheilSenRegressor(),
                HuberRegressor(),
                RidgeCV(),
    			LassoCV()
    kwargs:
    ----------
    image_name : str
        Image name (for display purposes).
    degree : int 
        The maximal degree of the polynomial features for sklearn.preprocessing.PolynomialFeatures. Default is 2.
    bins : int
        Binning size (in pixel units) for image binning. Default is 10.
    Analysis_ROIs : list of lists: [[left, right, top, bottom]]
        List of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the parabolic fit.
    calc_corr : boolean
        If True - the full image correction is calculated. Default is False.
    ignore_Y  : boolean
        If True - the polynomial fit to only X is performed. Default is False.
    linear_Y  : boolean
        If True - the polynomial fit to only X is perfromed, only linear variation along Y is allowed.
    Xsect : int
        X - coordinate for Y-cross-section.
    Ysect : int
        Y - coordinate for X-cross-section.
    disp_res : boolean
        Plot/ display the results. Defaults is True.
    save_res_png : boolean
        Save the analysis output into a PNG file. Default is False.
    save_correction_binary : boolean
        Save the image_name and img_correction_array data into a binary file. Default is False.
    res_fname : string
        Filename for the result image. Default is 'Image_Flattening.png'.
    label : string
        Optional image label.
    dpi : int
        Resolution (DPI) for the output PNG image.

    Returns:
    intercept, coefs, mse, img_correction_array
    '''
    image_name = kwargs.get("image_name", 'RawImageA')
    degree = kwargs.get("degree", 2)
    ysz, xsz = img.shape
    calc_corr = kwargs.get("calc_corr", False)
    ignore_Y = kwargs.get("ignore_Y", False)
    linear_Y = kwargs.get("linear_Y", False)
    Xsect = kwargs.get("Xsect", xsz//2)
    Ysect = kwargs.get("Ysect", ysz//2)
    disp_res = kwargs.get("disp_res", True)
    bins = kwargs.get("bins", 10) #bins = 10
    Analysis_ROIs = kwargs.get("Analysis_ROIs", [])
    save_res_png = kwargs.get("save_res_png", False)
    res_fname = kwargs.get("res_fname", 'Image_Flattening.png')
    dpi = kwargs.get("dpi", 300)

    # generate coefficient names
    data00 = pd.DataFrame({
        'x': np.random.randint(low=1, high=10, size=2),
        'y': np.random.randint(low=-1, high=1, size=2)})
    PolyFeats00 = PolynomialFeatures(degree=degree)
    dfPoly = pd.DataFrame(
        data=PolyFeats00.fit_transform(data00), 
        columns=PolyFeats00.get_feature_names_out(data00.columns))
    coeff_columns=', '.join(PolyFeats00.get_feature_names_out(data00.columns))
    # end of name generation
    
    img_binned = img[0:ysz//bins*bins, 0:xsz//bins*bins].astype(float).reshape(ysz//bins, bins, xsz//bins, bins).sum(3).sum(1)/bins/bins
    if len(Analysis_ROIs)>0:
            Analysis_ROIs_binned = [[ind//bins for ind in Analysis_ROI] for Analysis_ROI in Analysis_ROIs]
    else: 
        Analysis_ROIs_binned = []
    vmin, vmax = get_min_max_thresholds(img_binned, disp_res=False)
    yszb, xszb = img_binned.shape
    yb, xb = np.indices(img_binned.shape)
    if len(Analysis_ROIs_binned)>0:
        img_1D_list = []
        xb_1d_list = []
        yb_1d_list = []
        for Analysis_ROI_binned in Analysis_ROIs_binned:
            #Analysis_ROI : list of [left, right, top, bottom]
            img_1D_list = img_1D_list + img_binned[Analysis_ROI_binned[2]:Analysis_ROI_binned[3], Analysis_ROI_binned[0]:Analysis_ROI_binned[1]].ravel().tolist()
            xb_1d_list  = xb_1d_list + xb[Analysis_ROI_binned[2]:Analysis_ROI_binned[3], Analysis_ROI_binned[0]:Analysis_ROI_binned[1]].ravel().tolist()
            yb_1d_list  = yb_1d_list + yb[Analysis_ROI_binned[2]:Analysis_ROI_binned[3], Analysis_ROI_binned[0]:Analysis_ROI_binned[1]].ravel().tolist()
        img_1D = np.array(img_1D_list)
        xb_1d = np.array(xb_1d_list)
        yb_1d = np.array(yb_1d_list)
    else:
        img_1D = img_binned.ravel()
        xb_1d = xb.ravel()
        yb_1d = yb.ravel()

    img_binned_1D = img_binned.ravel()
    X_binned = np.vstack((xb.ravel(), yb.ravel())).T
    X = np.vstack((xb_1d, yb_1d)).T
    
    ysz, xsz = img.shape
    yf, xf = np.indices(img.shape)
    xf_1d = xf.ravel()/bins
    yf_1d = yf.ravel()/bins
    Xf = np.vstack((xf_1d, yf_1d)).T

    model = make_pipeline(PolynomialFeatures(degree), estimator)

    ymean = np.mean(yb_1d)
    model.fit(X, img_1D)
    '''
    if ignore_Y:
        ymean = np.mean(yb_1d)
        yb_1d_flat = yb_1d*0.0+ymean
        X_yflat = np.vstack((xb_1d, yb_1d_flat)).T
        model.fit(X_yflat, img_1D)

    else:
        model.fit(X, img_1D)
    '''

    model = make_pipeline(PolynomialFeatures(degree), estimator)
    model.fit(X, img_1D)
    if hasattr(model[-1], 'estimator_'):
        if ignore_Y:
            if degree == 2:
                # Estimator coefficients (1  x  y  x^2  x*y  y^2)
                model[-1].estimator_.coef_[0] = model[-1].estimator_.coef_[0] + model[-1].estimator_.coef_[2]*ymean + model[-1].estimator_.coef_[5]*ymean*ymean
                model[-1].estimator_.coef_[1] = model[-1].estimator_.coef_[1] + model[-1].estimator_.coef_[4]*ymean
                model[-1].estimator_.coef_[2] = 0.0
                model[-1].estimator_.coef_[4:6] = 0.0
            if degree == 4:
                # Estimator coeff. inds:  0  1  2  3    4    5    6    7      8      9    10   11      12       13    14
                # Estimator coefficients (1  x  y x^2  x*y  y^2  x^3  x^2*y  x*y^2  y^3  x^4  x^3*y  x^2*y^2  x*y^3  y^4)
                model[-1].estimator_.coef_[0] = model[-1].estimator_.coef_[0] + model[-1].estimator_.coef_[2]*ymean + model[-1].estimator_.coef_[5]*ymean*ymean
                + model[-1].estimator_.coef_[9]*ymean*ymean*ymean + model[-1].estimator_.coef_[14]*ymean*ymean*ymean*ymean
                model[-1].estimator_.coef_[1] = model[-1].estimator_.coef_[1] + model[-1].estimator_.coef_[4]*ymean + model[-1].estimator_.coef_[8]*ymean*ymean + model[-1].estimator_.coef_[13]*ymean*ymean*ymean
                model[-1].estimator_.coef_[3] = model[-1].estimator_.coef_[3] + model[-1].estimator_.coef_[7]*ymean + model[-1].estimator_.coef_[12]*ymean*ymean
                model[-1].estimator_.coef_[6] = model[-1].estimator_.coef_[6] + model[-1].estimator_.coef_[11]*ymean
                model[-1].estimator_.coef_[2] = 0.0
                model[-1].estimator_.coef_[4:6] = 0.0
                model[-1].estimator_.coef_[7:10] = 0.0
                model[-1].estimator_.coef_[11:] = 0.0
        else:
            if linear_Y:
                if degree == 2:
                    # Estimator coefficients (1  x  y  x^2  x*y  y^2)
                    model[-1].estimator_.coef_[2] = model[-1].estimator_.coef_[2] + model[-1].estimator_.coef_[5]*ymean
                    model[-1].estimator_.coef_[5] = 0.0
                if degree == 4:
                    # Estimator coeff. inds:  0  1  2  3    4    5    6    7      8      9    10   11      12       13    14
                    # Estimator coefficients (1  x  y x^2  x*y  y^2  x^3  x^2*y  x*y^2  y^3  x^4  x^3*y  x^2*y^2  x*y^3  y^4)
                    model[-1].estimator_.coef_[2] = model[-1].estimator_.coef_[2] + model[-1].estimator_.coef_[5]*ymean + model[-1].estimator_.coef_[9]*ymean*ymean + model[-1].estimator_.coef_[14]*ymean*ymean*ymean
                    model[-1].estimator_.coef_[4] = model[-1].estimator_.coef_[4] + model[-1].estimator_.coef_[8]*ymean + model[-1].estimator_.coef_[13]*ymean*ymean
                    model[-1].estimator_.coef_[7] = model[-1].estimator_.coef_[7] + model[-1].estimator_.coef_[12]*ymean
                    model[-1].estimator_.coef_[5] = 0.0
                    model[-1].estimator_.coef_[8] = 0.0
                    model[-1].estimator_.coef_[9] = 0.0
                    model[-1].estimator_.coef_[12] = 0.0
                    model[-1].estimator_.coef_[13] = 0.0
                    model[-1].estimator_.coef_[14] = 0.0
        coefs = model[-1].estimator_.coef_
        intercept = model[-1].estimator_.intercept_
    else:
        if ignore_Y:
            if degree==2:
                # Estimator coefficients (1  x  y  x^2  x*y  y^2)
                model[-1].coef_[0] = model[-1].coef_[0] + model[-1].coef_[2]*ymean + model[-1].coef_[5]*ymean*ymean
                model[-1].coef_[1] = model[-1].coef_[1] + model[-1].coef_[4]*ymean
                model[-1].coef_[2] = 0.0
                model[-1].coef_[4] = 0.0
                model[-1].coef_[5] = 0.0
            if degree == 4:
                # Estimator coeff. inds:  0  1  2  3    4    5    6    7      8      9    10   11      12       13    14
                # Estimator coefficients (1  x  y x^2  x*y  y^2  x^3  x^2*y  x*y^2  y^3  x^4  x^3*y  x^2*y^2  x*y^3  y^4)
                model[-1].coef_[0] = model[-1].coef_[0] + model[-1].coef_[2]*ymean + model[-1].coef_[5]*ymean*ymean
                + model[-1].coef_[9]*ymean*ymean*ymean + model[-1].coef_[14]*ymean*ymean*ymean*ymean
                model[-1].coef_[1] = model[-1].coef_[1] + model[-1].coef_[4]*ymean + model[-1].coef_[8]*ymean*ymean + model[-1].coef_[13]*ymean*ymean*ymean
                model[-1].coef_[3] = model[-1].coef_[3] + model[-1].coef_[7]*ymean + model[-1].coef_[12]*ymean*ymean
                model[-1].coef_[6] = model[-1].coef_[6] + model[-1].coef_[11]*ymean
                model[-1].coef_[2] = 0.0
                model[-1].coef_[4:6] = 0.0
                model[-1].coef_[7:10] = 0.0
                model[-1].coef_[11:] = 0.0
        else:
            if linear_Y:
                if degree == 2:
                    # Estimator coefficients (1  x  y  x^2  x*y  y^2)
                    model[-1].coef_[2] = model[-1].coef_[5] *ymean
                    model[-1].coef_[5] = 0.0
                if degree == 4:
                    # Estimator coeff. inds:  0  1  2  3    4    5    6    7      8      9    10   11      12       13    14
                    # Estimator coefficients (1  x  y x^2  x*y  y^2  x^3  x^2*y  x*y^2  y^3  x^4  x^3*y  x^2*y^2  x*y^3  y^4)
                    model[-1].coef_[2] = model[-1].coef_[2] + model[-1].coef_[5]*ymean + model[-1].coef_[9]*ymean*ymean + model[-1].coef_[14]*ymean*ymean*ymean
                    model[-1].coef_[4] = model[-1].coef_[4] + model[-1].coef_[8]*ymean + model[-1].coef_[13]*ymean*ymean
                    model[-1].coef_[7] = model[-1].coef_[7] + model[-1].coef_[12]*ymean
                    model[-1].coef_[5] = 0.0
                    model[-1].coef_[8] = 0.0
                    model[-1].coef_[9] = 0.0
                    model[-1].coef_[12] = 0.0
                    model[-1].coef_[13] = 0.0
                    model[-1].coef_[14] = 0.0
        coefs = model[-1].coef_
        intercept = model[-1].intercept_
    img_fit_1d = model.predict(X)
    scr = model.score(X, img_1D)
    mse = mean_squared_error(img_fit_1d, img_1D)
    img_fit = model.predict(X_binned).reshape(yszb, xszb)
    if calc_corr:
        img_correction_array = np.mean(img_fit_1d) / model.predict(Xf).reshape(ysz, xsz)
    else:
        img_correction_array = img * 0.0
        
    if disp_res:
        print('Estimator coefficients ' + coeff_columns + ' : ', coefs)
        print('Estimator intercept: ', intercept)
        
        fig, axs = plt.subplots(2,2, figsize = (12, 8))
        axs[0, 0].imshow(img_binned, cmap='Greys', vmin=vmin, vmax=vmax)
        axs[0, 0].grid(True)
        axs[0, 0].plot([Xsect//bins, Xsect//bins], [0, yszb], 'lime', linewidth = 0.5)
        axs[0, 0].plot([0, xszb], [Ysect//bins, Ysect//bins], 'cyan', linewidth = 0.5)
        if len(Analysis_ROIs_binned)>0:
            col_ROIs = 'yellow'
            axs[0, 0].text(0.3, 0.9, 'with Analysis ROIs', color=col_ROIs, transform=axs[0, 0].transAxes)
            for Analysis_ROI_binned in Analysis_ROIs_binned:
            #Analysis_ROI : list of [left, right, top, bottom]
                xi, xa, yi, ya = Analysis_ROI_binned
                ROI_patch = patches.Rectangle((xi,yi), np.abs(xa-xi)-2, np.abs(ya-yi)-2, linewidth=0.75, edgecolor=col_ROIs,facecolor='none')
                axs[0, 0].add_patch(ROI_patch)

        axs[0, 0].set_xlim((0, xszb))
        axs[0, 0].set_ylim((yszb, 0))
        axs[0, 0].set_title('{:d}-x Binned Raw:'.format(bins) + image_name)

        axs[0, 1].imshow(img_fit, cmap='Greys', vmin=vmin, vmax=vmax)
        axs[0, 1].grid(True)
        axs[0, 1].plot([Xsect//bins, Xsect//bins], [0, yszb], 'lime', linewidth = 0.5)
        axs[0, 1].plot([0, xszb], [Ysect//bins, Ysect//bins], 'cyan', linewidth = 0.5)
        axs[0, 1].set_xlim((0, xszb))
        axs[0, 1].set_ylim((yszb,0))
        axs[0, 1].set_title('{:d}-x Binned Fit: '.format(bins) + image_name)

        axs[1, 0].plot(img[Ysect, :],'b', label = image_name, linewidth =0.5)
        axs[1, 0].plot(xb[0,:]*bins, img_binned[Ysect//bins, :],'cyan', label = 'Binned '+ image_name)
        axs[1, 0].plot(xb[0,:]*bins, img_fit[Ysect//bins, :], 'yellow', linewidth=4, linestyle='--', label = 'Fit: '+ image_name)
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_xlabel('X-coordinate')

        axs[1, 1].plot(img[:, Xsect],'g', label = image_name, linewidth =0.5)
        axs[1, 1].plot(yb[:, 0]*bins, img_binned[:, Xsect//bins],'lime', label = 'Binned '+image_name)
        axs[1, 1].plot(yb[:, 0]*bins, img_fit[:, Xsect//bins], 'yellow', linewidth=4, linestyle='--', label = 'Fit: '+ image_name)
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_xlabel('Y-coordinate')
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   results saved into the file: '+res_fname)
    return intercept, coefs, mse, img_correction_array



##############################################
#      Two-Frame Image processing Functions
##############################################
def mutual_information_2d(x, y, sigma=1, bin=256, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    mi: float
        the computed similarity measure
    """
    bins = (bin, bin)

    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    #ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
    #                             output=jh)
    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))
    return mi


def Two_Image_NCC_SNR(img1, img2, **kwargs):
    '''
    Estimates normalized cross-correlation and SNR of two images.
    ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    
    Calculates SNR from cross-correlation of two images after [1, 2, 3].
    
    Parameters
    ---------
    img1 : 2D array
    img2 : 2D array
     
    kwargs:
    zero_mean: boolean
        if True, cross-correlation is zero-mean
    
    Returns:
        NCC, SNR : float, float
            NCC - normalized cross-correlation coefficient
            SNR - Signal-to-Noise ratio based on NCC
    
   [1] J. Frank, L. AI-Ali, Signal-to-noise ratio of electron micrographs obtained by cross correlation. Nature 256, 4 (1975).
   [2] J. Frank, in: Computer Processing of Electron Microscopic Images. Ed. P.W. Hawkes (Springer, Berlin, 1980).
   [3] M. Radermacher, T. Ruiz, On cross-correlations, averages and noise in electron microscopy. Acta Crystallogr. Sect. F Struct. Biol. Commun. 75, 12–18 (2019).
    
    '''
    zero_mean = kwargs.get("zero_mean", True)
    
    # convert into floats
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    if img1.shape==img2.shape:
        ysz, xsz = img1.shape
        if zero_mean:
            img1 = img1- img1.mean()
            img2 = img2- img2.mean()
        xy = np.sum((img1.ravel()*img2.ravel()))/(xsz*ysz)
        xx = np.sum((img1.ravel()*img1.ravel()))/(xsz*ysz)
        yy = np.sum((img2.ravel()*img2.ravel()))/(xsz*ysz)
        NCC = xy / np.sqrt(xx*yy)
        SNR = NCC / (1-NCC)

    else:
        print("img1 and img2 shapes must be equal")
        NCC = 0.0
        SNR = 0.0
    
    return NCC, SNR

def Two_Image_FSC(img1, img2, **kwargs):
    '''
    Performs Fourier Shell Correlation to determine the image resolution, after [1]. ©G.Shtengel, 10/2019. gleb.shtengel@gmail.com
    FSC is determined from radially averaged foirier cross-correlation (with optional selection of range of angles for radial averaging).
    
    Parameters
    ---------
    img1 : 2D array
    img2 : 2D array
     
    kwargs:
    SNRt : float
        SNR threshold for determining the resolution bandwidth
    fit_data : boolean
        If True the BW will be extracted fron inverse power fit.
        If False, the BW will be extracted from the smoothed FSC data
    smooth_aperture : int
        Smoothing aperture. Default is 20.
    fit_power : int
        parameter for FSC data fitting: FSC_fit  = a/(x**fit_power+a)
    fr_cutoff : float
        The fractional value between 0.0 and 1.0. The data points within the frequency range [0 : max_frequency*cutoff]  will be used.
    astart : float
        Start angle for radial averaging. Default is 0.0
    astop : float
        Stop angle for radial averaging. Default is 90.0
    symm : int
        Symmetry factor (how many times Start and stop angle intervalks are repeated within 360 deg). Default is 4.
    disp_res : boolean
        display results (plots) (default is False)
    ax : axis object (matplotlip)
        to export the plot
    save_res_png : boolean
        save results into PNG file (default is False)
    res_fname : string
        filename for the result image ('SNR_result.png')
    img_labels : [string, string]
        optional image labels
    dpi : int
        dots-per-inch resolution for the output image
    pixel : float
        optional pixel size in nm. If not provided, will be ignored.
        if provided, second axis will be added on top with inverse pixels
    xrange : [float, float]
        range of x axis in FSC plot in inverse pixels
        if not provided [0, 0.5] range will be used
    verbose : boolean
        print the outputs. Default is False

    
    Returns FSC_sp_frequencies, FSC_data, x2, T, FSC_bw
        FSC_sp_frequencies : float array 
            Spatial Frequency (/Nyquist) - for FSC plot
        FSC_data: float array
        x2 : float array
            Spatial Frequency (/Nyquist) - for threshold line plot
        T : float array
            threshold line plot
        FSC_bw : float
            the value of FSC determined as an intersection of smoothed data threshold
        
    [1]. M. van Heela, and M. Schatzb, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151, 250-262 (2005)
    '''   
    SNRt = kwargs.get("SNRt", 0.1)
    smooth_aperture = kwargs.get("smooth_aperture", 20)
    fit_data = kwargs.get("fit_data", False)
    fit_power = kwargs.get("fit_power", 3)
    fr_cutoff = kwargs.get("fr_cutoff", 0.9)
    astart = kwargs.get("astart", 0.0)
    astop = kwargs.get("astop", 90.0)
    symm = kwargs.get("symm", 4)
    disp_res = kwargs.get("disp_res", False)
    ax = kwargs.get("ax", '')
    save_res_png = kwargs.get("save_res_png", False)
    res_fname = kwargs.get("res_fname", 'FSC_results.png')
    img_labels = kwargs.get("img_labels", ['Image 1', 'Image 2'])
    dpi = kwargs.get("dpi", 300)
    pixel = kwargs.get("pixel", 0.0)
    verbose = kwargs.get('verbose', False)
    cmap = kwargs.get('cmap', 'Greys_r')
    xrange = kwargs.get("xrange", [0, 0.5])
    
    #Check whether the inputs dimensions match and the images are square
    if disp_res:
        if ( np.shape(img1) != np.shape(img2) ) :
            print('input images must have the same dimensions')
        if ( np.shape(img1)[0] != np.shape(img1)[1]) :
            print('input images must be squares')
    I1 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img1)))  # I1 and I2 store the FFT of the images to be used in the calcuation for the FSC
    I2 = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img2)))

    C_imre = np.multiply(I1,np.conj(I2))
    C12_ar = np.abs(np.multiply((I1+I2),np.conj(I1+I2)))
    y0,x0 = argmax2d(C12_ar)
    C1 = radial_profile_select_angles(np.abs(np.multiply(I1,np.conj(I1))), [x0,y0], astart=astart, astop=astop, symm=symm)
    C2 = radial_profile_select_angles(np.abs(np.multiply(I2,np.conj(I2))), [x0,y0], astart=astart, astop=astop, symm=symm)
    C  = radial_profile_select_angles(np.real(C_imre), [x0,y0], astart=astart, astop=astop, symm=symm) + 1j * radial_profile_select_angles(np.imag(C_imre), [x0,y0], astart=astart, astop=astop, symm=symm)

    FSC_data = np.abs(C)/np.sqrt(np.abs(np.multiply(C1,C2)))
    '''
    T is the SNR threshold calculated accoring to the input SNRt, if nothing is given
    a default value of 0.1 is used.
    
    x2 contains the normalized spatial frequencies
    '''
    r = np.arange(1+np.shape(img1)[0])
    n = 2*np.pi*r
    n[0] = 1
    eps = np.finfo(float).eps
    t1 = np.divide(np.ones(np.shape(n)),n+eps)
    t2 = SNRt + 2*np.sqrt(SNRt)*t1 + np.divide(np.ones(np.shape(n)),np.sqrt(n))
    t3 = SNRt + 2*np.sqrt(SNRt)*t1 + 1
    T = np.divide(t2,t3)
    #FSC_sp_frequencies = np.arange(np.shape(C)[0])/(np.shape(img1)[0]/np.sqrt(2.0))
    #x2 = r/(np.shape(img1)[0]/np.sqrt(2.0))
    FSC_sp_frequencies = np.arange(np.shape(C)[0])/(np.shape(img1)[0])
    x2 = r/(np.shape(img1)[0])
    FSC_data_smooth = smooth(FSC_data, smooth_aperture)
    FSC_bw, fr_fit, FSC_fit, fitOK = find_BW(FSC_sp_frequencies, FSC_data_smooth,
                              SNRt = SNRt,
                              verbose = verbose,
                              fit_data = fit_data,
                              fit_power = fit_power,
                              fr_cutoff = fr_cutoff)
    if fitOK > 0:
        if verbose:
                print('Cannot determine BW accurately: not enough points')
    '''
    If the disp_res input is set to True, an output plot is generated. 
    '''
    if disp_res:
        if ax=='':
            fig = plt.figure(figsize=(8,12))
            axs0 = fig.add_subplot(3,2,1)
            axs1 = fig.add_subplot(3,2,2)
            axs2 = fig.add_subplot(3,2,3)
            axs3 = fig.add_subplot(3,2,4)
            ax = fig.add_subplot(3,1,3)
            fig.subplots_adjust(left=0.01, bottom=0.06, right=0.99, top=0.975, wspace=0.25, hspace=0.10)
            vmin1, vmax1 = get_min_max_thresholds(img1, disp_res=False)
            vmin2, vmax2 = get_min_max_thresholds(img2, disp_res=False)
            axs0.imshow(img1, cmap=cmap, vmin=vmin1, vmax=vmax1)
            axs1.imshow(img2, cmap=cmap, vmin=vmin2, vmax=vmax2)
            x = np.linspace(0, 1.41, 500)
            axs2.set_xlim(-1,1)
            axs2.set_ylim(-1,1)
            axs2.imshow(np.log(np.abs(I1)), extent=[-1, 1, -1, 1], cmap = 'Greys_r')
            axs3.set_xlim(-1,1)
            axs3.set_ylim(-1,1)
            axs3.imshow(np.log(np.abs(I2)), extent=[-1, 1, -1, 1], cmap = 'Greys_r')
            for i in np.arange(symm):
                ai = np.radians(astart + 360.0/symm*i)
                aa = np.radians(astop + 360.0/symm*i)
                axs2.plot(x * np.cos(ai), x * np.sin(ai), color='orange', linewidth = 0.5)
                axs3.plot(x * np.cos(ai), x * np.sin(ai), color='orange', linewidth = 0.5)
                axs2.plot(x * np.cos(aa), x * np.sin(aa), color='orange', linewidth = 0.5)
                axs3.plot(x * np.cos(aa), x * np.sin(aa), color='orange', linewidth = 0.5)
            ttls = img_labels.copy()
            ttls.append('FFT of '+img_labels[0])
            ttls.append('FFT of '+img_labels[1])
            for axi, ttl in zip([axs0, axs1, axs2, axs3], ttls):
                axi.grid(False)
                axi.axis(False)
                axi.set_title(ttl)

    if disp_res or ax != '':
        ax.plot(FSC_sp_frequencies, FSC_data, label = 'FSC data', color='r')
        ax.plot(FSC_sp_frequencies, FSC_data_smooth, label = 'FSC data smoothed', color='b')
        if pixel>1e-6:
            label = 'FSC BW = {:.3f} inv.pix., or {:.2f} nm'.format(FSC_bw, pixel/FSC_bw)
        else:
            label = 'FSC BW = {:.3f}'.format(FSC_bw)
        ax.plot(np.array((FSC_bw,FSC_bw)), np.array((0.0,1.0)), '--', label = label, color = 'g')
        if fit_data:
            ax.plot(fr_fit, FSC_fit, label = 'a/(x**{:d}+a) Fit'.format(fit_power), linewidth=1, color='g')
            ax.plot(fr_fit, fr_fit*0.0+SNRt, '--', label = 'Threshold SNR = {:.3f}'.format(SNRt), color='m')
        else:
            ax.plot(FSC_sp_frequencies, FSC_sp_frequencies*0.0+SNRt, '--', label = 'Threshold SNR = {:.3f}'.format(SNRt), color='m')

        ax.plot([FSC_bw, FSC_bw], [0, 1], '--', label = 'BW = {:.3f}'.format(FSC_bw), color='brown')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.set_xlim(xrange)
        ax.legend()
        ax.set_xlabel('Spatial Frequency (inverse pixels)')
        ax.set_ylabel('FSC Magnitude')
        ax.grid(True)
        if pixel>1e-6:
            
            def forward(x):
                return x/pixel
            def inverse(x):
                return x*pixel
            secax = ax.secondary_xaxis('top', functions=(forward, inverse))
            secax.set_xlabel('Spatial Frequency ($nm^{-1}$)') 
        
    if disp_res:
        ax.set_position([0.1, 0.05, 0.85, 0.28])
        print('FSC BW = {:.5f}'.format(FSC_bw))
        if save_res_png:
            fig.savefig(res_fname, dpi=dpi)
            print('Saved the results into the file: ', res_fname)
    return (FSC_sp_frequencies, FSC_data, x2, T, FSC_bw)


def Two_Image_Analysis(params):
    '''
    Analyzes the registration  quality between two frames (for DASK registration analysis)

    Parameterss:
    params : list of params
        params = [frame1_filename, frame2_filename, eval_bounds, eval_metrics]
        eval_bounds = [xi,  xa, yi, ya]
        eval_metrics = ['NSAD', 'NCC', 'NMI', 'FSC']

    Returns
        results : list of results
    '''
    frame1_filename, frame2_filename, eval_bounds, eval_metrics = params
    xi_eval,  xa_eval, yi_eval, ya_eval = eval_bounds


    I1 = tiff.imread(os.path.normpath(frame1_filename))
    I1c = I1[yi_eval:ya_eval, xi_eval:xa_eval].astype(float)
    I2 = tiff.imread(os.path.normpath(frame2_filename))
    I2c = I2[yi_eval:ya_eval, xi_eval:xa_eval].astype(float)
    fr_mean = np.abs(I1c/2.0 + I2c/2.0)
    dy, dx = np.shape(I2c)

    results = []
    for metric in eval_metrics:
        if metric == 'NSAD':
            try:
                nsad = np.mean(np.abs(I1c-I2c))/(np.mean(fr_mean)-np.amin(fr_mean))
            except:
                nsad = 0.0
            results.append(nsad)
        if metric == 'NCC':
            try:
                ncc = Two_Image_NCC_SNR(I1c, I2c)[0]
            except:
                ncc = 0.0
            results.append(ncc)
        if metric == 'NMI':
            try:
                nmi = mutual_information_2d(I1c.ravel(), I2c.ravel(), sigma=1.0, bin=2048, normalized=True)
            except:
                nmi = 0.0
            results.append(nmi)
        if metric == 'FSC':
            #SNRt is SNR threshold for determining the resolution bandwidth
            # force square images for FSC
            try:
                if dx != dy:
                    d=min((dx//2, dy//2))
                    fsc = Two_Image_FSC(I1c[dy//2-d:dy//2+d, dx//2-d:dx//2+d], I2c[dy//2-d:dy//2+d, dx//2-d:dx//2+d], SNRt=0.143, disp_res=False)[4]
                else:
                    fsc = Two_Image_FSC(I1c, I2c, SNRt=0.143, disp_res=False)[4]
            except:
                fsc = 0.00
            results.append(fsc)

    return results



##########################################
#         MRC stack analysis functions
##########################################

def evaluate_registration_two_frames(params_mrc):
    '''
    Helper function used by DASK routine. Analyzes registration between two frames.
    ©G.Shtengel, 10/2020. gleb.shtengel@gmail.com

    Parameters:
    params_mrc : list of mrc_filename, fr, evals, save_frame_png, filename_frame_png
    mrc_filename  : string
        full path to mrc filename
    fr : int
        Index of the SECOND frame
    evals :  list of image bounds to be used for evaluation exi_eval, xa_eval, yi_eval, ya_eval 


    Returns:
    image_nsad, image_ncc, image_mi   : float, float, float

    '''
    mrc_filename, fr, invert_data, evals, save_frame_png, filename_frame_png = params_mrc
    mrc_filename  = os.path.normpath(mrc_filename)
    filename_frame_png = os.path.normpath(filename_frame_png)
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    '''
    0 8-bit signed integer (range -128 to 127)
    1 16-bit signed integer
    2 32-bit signed real
    3 transform : complex 16-bit integers
    4 transform : complex 32-bit reals
    6 16-bit unsigned integer
    12 16-bit float (IEEE754)
    101 4-bit data packed two per byte
    
    mode 0 -> uint8
    mode 1 -> int16
    mode 2 -> float32
    mode 4 -> complex64
    mode 6 -> uint16
    '''
    mrc_mode = mrc_obj.header.mode
    if mrc_mode==0:
        dt_mrc=np.uint8
    if mrc_mode==1:
        dt_mrc=np.int16
    if mrc_mode==2:
        dt_mrc=np.float32
    if mrc_mode==4:
        dt_mrc=np.complex64
    if mrc_mode==6:
        dt_mrc=np.uint16

    xi_eval, xa_eval, yi_eval, ya_eval = evals
    if invert_data:
        prev_frame = -1.0 * (((mrc_obj.data[fr-1, yi_eval:ya_eval, xi_eval:xa_eval]).astype(dt_mrc)).astype(float))
        curr_frame = -1.0 * (((mrc_obj.data[fr, yi_eval:ya_eval, xi_eval:xa_eval]).astype(dt_mrc)).astype(float))
    else:
        prev_frame = (mrc_obj.data[fr-1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float)
        curr_frame = (mrc_obj.data[fr, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float)
    fr_mean = np.abs(curr_frame/2.0 + prev_frame/2.0)
    image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)-np.amin(fr_mean))
    #image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean))
    image_ncc = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
    image_mi = mutual_information_2d(prev_frame.ravel(), curr_frame.ravel(), sigma=1.0, bin=2048, normalized=True)

    if save_frame_png:
        fr_img = (mrc_obj.data[fr, :, :].astype(dt_mrc)).astype(float)
        yshape, xshape = fr_img.shape
        fig, ax = plt.subplots(1,1, figsize=(3.0*xshape/yshape, 3))
        fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
        dmin, dmax = get_min_max_thresholds(fr_img[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
        if invert_data:
            ax.imshow(fr_img, cmap='Greys_r', vmin=dmin, vmax=dmax)
        else:
            ax.imshow(fr_img, cmap='Greys', vmin=dmin, vmax=dmax)
        #ax.text(0.06, 0.95, 'Frame={:d},  NSAD={:.3f},  NCC={:.3f},  NMI={:.3f}'.format(fr, image_nsad, image_ncc, image_mi), color='red', transform=ax.transAxes, fontsize=12)
        ax.text(0.02, 0.95, 'Frame={:d}'.format(fr), color='red', transform=ax.transAxes, fontsize=11)
        ax.text(0.50, 0.95, 'NSAD={:.3f}'.format(image_nsad), color='red', transform=ax.transAxes, fontsize=11)
        ax.text(0.50, 0.85, 'NCC={:.3f}'.format(image_ncc), color='red', transform=ax.transAxes, fontsize=11)
        ax.text(0.50, 0.75, 'NMI={:.3f}'.format(image_mi), color='red', transform=ax.transAxes, fontsize=11)
        rect_patch = patches.Rectangle((xi_eval, yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2, linewidth=1.0, edgecolor='yellow',facecolor='none')
        ax.add_patch(rect_patch)
        ax.axis('off')
        fig.savefig(filename_frame_png, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)   # save the figure to file
        plt.close(fig)

    mrc_obj.close()
    return image_nsad, image_ncc, image_mi


def analyze_mrc_stack_registration(mrc_filename, **kwargs):
    '''
    Read MRC stack and analyze registration - calculate NSAD, NCC, and MI.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters:
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed.
    kwargs:
    ---------
    DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames.
    invert_data : boolean
        If True, the data will be inverted.
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check.
    save_filename : str
        Path to the filename to save the results. If empty, mrc_filename+'_RegistrationQuality.xlsx' will be used.
    save_sample_frames_png : boolean
        If True, sample frames with superimposed eval box and registration analysis data will be saved into png files. Default is True.

    Returns:
    reg_summary : PD data frame, registration_summary_xlsx : path to summary XLSX spreadsheet file
    '''
    mrc_filename  = os.path.normpath(mrc_filename)

    Sample_ID = kwargs.get("Sample_ID", '')
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    use_DASK, status_update_address = check_DASK(DASK_client)

    invert_data =  kwargs.get("invert_data", False)
    save_res_png  = kwargs.get("save_res_png", True )
    save_filename = kwargs.get("save_filename", mrc_filename )
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    registration_summary_xlsx = save_filename.replace('.mrc', '_RegistrationQuality.xlsx')
    save_sample_frames_png = kwargs.get("save_sample_frames_png", True)

    if sliding_evaluation_box:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will use sliding (linearly) evaluation box')
        print('   Starting with box:  ', start_evaluation_box)
        print('   Finishing with box: ', stop_evaluation_box)
    else:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will use fixed evaluation box: ', evaluation_box)

    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    mrc_mode = header.mode
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    header_dict = {}
    for record in header.dtype.names: # create dictionary from the header data
        if ('extra' not in record) and ('label' not in record):
            header_dict[record] = header[record]
    '''
    0 8-bit signed integer (range -128 to 127)
    1 16-bit signed integer
    2 32-bit signed real
    3 transform : complex 16-bit integers
    4 transform : complex 32-bit reals
    6 16-bit unsigned integer
    12 16-bit float (IEEE754)
    101 4-bit data packed two per byte
    
    mode 0 -> uint8
    mode 1 -> int16
    mode 2 -> float32
    mode 4 -> complex64
    mode 6 -> uint16
    '''
    if mrc_mode==0:
        dt_mrc=np.uint8
    if mrc_mode==1:
        dt_mrc=np.int16
    if mrc_mode==2:
        dt_mrc=np.float32
    if mrc_mode==4:
        dt_mrc=np.complex64
    if mrc_mode==6:
        dt_mrc=np.uint16
    print('mrc_mode={:d} '.format(mrc_mode), ', dt_mrc=', dt_mrc)

    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny
    evals = [xi_eval, xa_eval, yi_eval, ya_eval]
    
    frame_inds_default = np.arange(nz-1)+1
    frame_inds = np.array(kwargs.get("frame_inds", frame_inds_default))
    nf = frame_inds[-1]-frame_inds[0]+1
    if frame_inds[0]==0:
        frame_inds = frame_inds+1
    sample_frame_inds = [frame_inds[nf//10], frame_inds[nf//2], frame_inds[nf//10*9]]
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will analyze regstrations in {:d} frames'.format(len(frame_inds)))
    print('Will save the data into ' + registration_summary_xlsx)
    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0
    
    params_mrc_mult = []
    xi_evals = np.zeros(nf, dtype=np.int16)
    xa_evals = np.zeros(nf, dtype=np.int16)
    yi_evals = np.zeros(nf, dtype=np.int16)
    ya_evals = np.zeros(nf, dtype=np.int16)
    for j, fr in enumerate(frame_inds):
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*(fr-frame_inds[0])//nf
            yi_eval = start_evaluation_box[0] + dy_eval*(fr-frame_inds[0])//nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
            evals = [xi_eval, xa_eval, yi_eval, ya_eval]
        xi_evals[j] = xi_eval
        xa_evals[j] = xa_eval
        yi_evals[j] = yi_eval
        ya_evals[j] = ya_eval
        if fr in sample_frame_inds:
            save_frame_png = save_sample_frames_png
            filename_frame_png = os.path.splitext(save_filename)[0]+'_sample_image_frame{:d}.png'.format(fr)
        else:
            save_frame_png = False
            filename_frame_png = os.path.splitext(save_filename)[0]+'_sample_image_frame.png'
        params_mrc_mult.append([mrc_filename, fr, invert_data, evals, save_frame_png, filename_frame_png])
        
    if use_DASK:
        mrc_obj.close()
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
        futures = DASK_client.map(evaluate_registration_two_frames, params_mrc_mult, retries = DASK_client_retries)
        dask_results = DASK_client.gather(futures)
        image_nsad = np.array([res[0] for res in dask_results])
        image_ncc = np.array([res[1] for res in dask_results])
        image_mi = np.array([res[2] for res in dask_results])
    else:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using Local Computation')
        image_nsad = np.zeros(nf, dtype=float)
        image_ncc = np.zeros(nf, dtype=float)
        image_mi = np.zeros(nf, dtype=float)
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*frame_inds[0]//nf
            yi_eval = start_evaluation_box[0] + dy_eval*frame_inds[0]//nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
        if invert_data:
            prev_frame = -1.0 * ((mrc_obj.data[frame_inds[0]-1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float))
        else:
            prev_frame =(mrc_obj.data[frame_inds[0]-1, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float)
        for j, frame_ind in enumerate(tqdm(frame_inds, desc='Evaluating frame registration: ')):
            if sliding_evaluation_box:
                xi_eval = start_evaluation_box[2] + dx_eval*j//nf
                yi_eval = start_evaluation_box[0] + dy_eval*j//nf
                if start_evaluation_box[3] > 0:
                    xa_eval = xi_eval + start_evaluation_box[3]
                else:
                    xa_eval = nx
                if start_evaluation_box[1] > 0:
                    ya_eval = yi_eval + start_evaluation_box[1]
                else:
                    ya_eval = ny
            
            if invert_data:
                curr_frame = -1.0 * ((mrc_obj.data[frame_ind, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float))
            else:
                curr_frame = (mrc_obj.data[frame_ind, yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float)

            fr_mean = np.abs(curr_frame/2.0 + prev_frame/2.0)

            image_ncc[j-1] = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]

            image_nsad[j-1] =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)-np.amin(fr_mean))
            image_mi[j-1] = mutual_information_2d(prev_frame.ravel(), curr_frame.ravel(), sigma=1.0, bin=2048, normalized=True)
            prev_frame = curr_frame.copy()
            if (frame_ind in sample_frame_inds) and save_sample_frames_png:
                filename_frame_png = os.path.splitext(save_filename)[0]+'_sample_image_frame{:d}.png'.format(j)
                fr_img = (mrc_obj.data[frame_ind, :, :].astype(dt_mrc)).astype(float)
                yshape, xshape = fr_img.shape
                fig, ax = plt.subplots(1,1, figsize=(3.0*xshape/yshape, 3))
                fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
                dmin, dmax = get_min_max_thresholds(fr_img[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
                if invert_data:
                    ax.imshow(fr_img, cmap='Greys_r', vmin=dmin, vmax=dmax)
                else:
                    ax.imshow(fr_img, cmap='Greys', vmin=dmin, vmax=dmax)
                ax.text(0.02, 0.95, 'Frame={:d}'.format(frame_ind), color='red', transform=ax.transAxes, fontsize=11)
                ax.text(0.50, 0.95, 'NSAD={:.3f}'.format(image_nsad[j-1]), color='red', transform=ax.transAxes, fontsize=11)
                ax.text(0.50, 0.85, 'NCC={:.3f}'.format(image_ncc[j-1]), color='red', transform=ax.transAxes, fontsize=11)
                ax.text(0.50, 0.75, 'NMI={:.3f}'.format(image_mi[j-1]), color='red', transform=ax.transAxes, fontsize=11)
                rect_patch = patches.Rectangle((xi_eval, yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2, linewidth=1.0, edgecolor='yellow',facecolor='none')
                ax.add_patch(rect_patch)
                ax.axis('off')
                fig.savefig(filename_frame_png, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)   # save the figure to file
                plt.close(fig)

        mrc_obj.close()
    
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)] 
    #image_ncc = image_ncc[1:-1]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_mi), np.median(image_mi), np.std(image_mi)]

    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the Registration Quality Statistics into the file: ', registration_summary_xlsx)
    xlsx_writer = pd.ExcelWriter(registration_summary_xlsx, engine='xlsxwriter')
    columns=['Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval', 'NSAD', 'NCC', 'NMI']
    reg_summary = pd.DataFrame(np.vstack((frame_inds, xi_evals, xa_evals, yi_evals, ya_evals, image_nsad, image_ncc, image_mi)).T, columns = columns, index = None)
    reg_summary.to_excel(xlsx_writer, index=None, sheet_name='Registration Quality Statistics')
    Stack_info = pd.DataFrame([{'Stack Filename' : mrc_filename, 'Sample_ID' : Sample_ID, 'invert_data' : invert_data}]).T # prepare to be save in transposed format
    header_info = pd.DataFrame([header_dict]).T
    #Stack_info = Stack_info.append(header_info)  append has been removed from pandas as of 2.0.0, use concat instead
    Stack_info = pd.concat([Stack_info, header_info], axis=1)
    Stack_info.to_excel(xlsx_writer, header=False, sheet_name='Stack Info')
    #xlsx_writer.save()
    xlsx_writer.close()

    return reg_summary, registration_summary_xlsx


def show_eval_box_mrc_stack(mrc_filename, **kwargs):
    '''
    Read MRC stack and display the eval box for each frame from the list.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
     
    kwargs:
    frame_inds : array
        List of frame indices to display the evaluation box. If not provided, three frames will be used:
        [nz//10,  nz//2, nz//10*9] where nz is number of frames in mrc stack
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above.
    box_linewidth : float
        linewidth for the box outline. Default is 1.0.
    box_color : color
        color for the box outline. Default is yellow.
    invert_data : Boolean
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    ax : matplotlib ax artist.
        if provided, the data is exported to external ax object.
    '''
    mrc_filename  = os.path.normpath(mrc_filename)

    Sample_ID = kwargs.get("Sample_ID", '')
    save_res_png  = kwargs.get("save_res_png", True )
    save_filename = kwargs.get("save_filename", mrc_filename )
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    box_linewidth = kwargs.get("box_linewidth", 1.0)
    box_color = kwargs.get("box_color", 'yellow')
    invert_data =  kwargs.get("invert_data", False)
    ax = kwargs.get("ax", '')
    plot_internal = (ax == '')

    mrc = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc.header
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    '''
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    '''
    mrc_mode = header.mode
    if mrc_mode==0:
        dt_mrc=np.uint8
    if mrc_mode==1:
        dt_mrc=np.int16
    if mrc_mode==2:
        dt_mrc=np.float32
    if mrc_mode==4:
        dt_mrc=np.complex64
    if mrc_mode==6:
        dt_mrc=np.uint16

    frame_inds = kwargs.get("frame_inds", [nz//10,  nz//2, nz//10*9] )
    
    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny

    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    for fr_ind in frame_inds: 
        eval_frame = (mrc.data[fr_ind, :, :].astype(dt_mrc)).astype(float)

        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*fr_ind//nz
            yi_eval = start_evaluation_box[0] + dy_eval*fr_ind//nz
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny

        if plot_internal:
            fig, ax = plt.subplots(1,1, figsize = (10.0, 11.0*ny/nx))
        dmin, dmax = get_min_max_thresholds(eval_frame[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
        if invert_data:
            ax.imshow(eval_frame, cmap='Greys_r', vmin=dmin, vmax=dmax)
        else:
            ax.imshow(eval_frame, cmap='Greys', vmin=dmin, vmax=dmax)
        ax.grid(True, color = "cyan")
        ax.set_title(Sample_ID + ' '+mrc_filename +',  frame={:d}'.format(fr_ind))
        rect_patch = patches.Rectangle((xi_eval,yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2,
            linewidth=box_linewidth, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect_patch)
        if save_res_png  and plot_internal:
            fname = os.path.splitext(save_filename)[0] + '_frame_{:d}_evaluation_box.png'.format(fr_ind)
            fig.savefig(fname, dpi=300)

    mrc.close()


def plot_cross_sections_mrc_stack(mrc_filename, **kwargs):
    '''
    Read MRC stack and plot the ortho cross-sections.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com

    Parameters:
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
     
    kwargs:
    ---------
    XZ_section : boolean
        If True (default), XZ cros-section is present.
    ZY_section : boolean
        If True (default), ZY cros-section is present.
    voxel_size : array or list of 3 floats
        Voxel size (x, y, z) in um. Default is the data obtrained from cellA data in the MRC file.
    center_coordinates : array or list of 3 floats
        Center coordinates (x, y, z) in um. Default is middle of the stack.
    box_dimensions : array or list of 3 floats
        Dimensions of the sections to plot (x, y, z) in um (box is centered around center coordinates). Default is full stack size.
    xsection_offsets  : array or list of 3 floats
        Offsets for cross-section location from the center of the box (x, y, z) in um. Default is [0.0, 0.0, 0.0].
    addtl_sp : float
        Additional white space between the cross-section plots. Default is 0.0.
    xsection_linewidth : float
        Width of the cross-section line. Default is 0.5.
    xsection_line_color : string
        Color of the cross-secion line. Default is 'white'.
    EM_min : float
        Min value for EM data range. If not defined will be determined automatically from the section data.
    EM_max : float
        Max value for EM data range. If not defined will be determined automatically from the section data.
    display_scale_bars : boolean
        If True (default), the scale bars are displayed in cross-sections.
    loc : (float, float)
        Bar location in fractional axis coordinates (0, 0) is left bottom corner. (1, 1) is top right corner.
        Default is (0.1, 0.9)
    bar_length_um : float
        Length of the scale bar (um). Default is 1um.
    bar_width : float
        Width of the scale bar. Defalt is 5.0.
    bar_color : string
        Color of the scale bar. Defalt is 'white'.
    display_scale_bar_labels : boolean
        If True (default), the scale bar labels are displayed.
    label : string
        Scale bar label. Default is length in um.
    label_color : color
        Color of the scale bar label. Default is the same as bar_color.
    label_font_size : int
        Font Size of the scale bar label. Default is 12.
    label_offset : int
        Additional vertical offset for the label position. Default is 0.
    save_PNG : boolean
        If True (default), the data will be saved into PNG file.
    save_filename : string
        Filename to save the image. Defaults is mrc_filename.replace('.mrc', '_crosssections.png')
    dpi : int
        DPI for the PNG file. Dafult is 300.
    
    Returns: images, axs
    '''
    display_scale_bars = kwargs.get('display_scale_bars', True)
    bar_length_um = kwargs.get('bar_length_um', 1.0)  #in um
    loc = kwargs.get('loc', (0.1, 0.9))
    bar_width = kwargs.get('bar_width', 5.0)
    bar_color = kwargs.get('bar_color', 'white')
    display_scale_bar_labels = kwargs.get('display_scale_bar_labels', True)
    bar_label = kwargs.get('bar_label', '{:.1f} μm'.format(bar_length_um))
    label_color = kwargs.get('label_color', bar_color)
    label_font_size = kwargs.get('label_font_size', 12)
    label_offset = kwargs.get('label_offset', 0)
    bar_kwargs = {'bar_length_um' : bar_length_um,
                 'bar_width' : bar_width,
                 'bar_color' : bar_color,
                 'loc' : loc,
                 'display_scale_bar_labels' : display_scale_bar_labels,
                 'bar_label' : bar_label,
                 'label_color' : label_color,
                 'label_font_size' : label_font_size,
                 'label_offset' : label_offset}
    xsection_linewidth = kwargs.get('xsection_linewidth', 0.5)
    xsection_line_color = kwargs.get('xsection_line_color', 'white')
    addtl_sp = kwargs.get('addtl_sp', 0.0)
    XZ_section = kwargs.get('XZ_section', True)
    ZY_section = kwargs.get('ZY_section', True)

    mrc_filename  = os.path.normpath(mrc_filename)
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    header = mrc_obj.header
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    voxel_size_angstr = mrc_obj.voxel_size # Angstrom per pixel
    voxel_size = np.array(kwargs.get('voxel_size', [voxel_size_angstr.x/1.0e4, voxel_size_angstr.y/1.0e4, voxel_size_angstr.z/1.0e4])) # in um per pixel
    stack_size = np.array([nx*voxel_size[0], ny*voxel_size[1], nz*voxel_size[2]])
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   EM cross-sections dimensions (um):', stack_size)
    cc_default = stack_size/2.0
    
    center_coordinates = np.array(kwargs.get('center_coordinates', cc_default))   # in um
    print('Center coordinates (um):', center_coordinates)
    box_dimensions = np.array(kwargs.get('box_dimensions', stack_size))           # in um
    xsection_offsets = np.array(kwargs.get('xsection_offsets', [0.0, 0.0, 0.0]))                    # in um
        
    save_PNG = kwargs.get('save_PNG', True)
    save_filename = kwargs.get('save_filename', mrc_filename.replace('.mrc', '_crosssections.png'))
    dpi = kwargs.get('dpi', 300)
    
    # Build XYZ ranges 
    cc_EM = center_coordinates / voxel_size
    cs_EM = (center_coordinates + xsection_offsets) / voxel_size
    bd_EM = box_dimensions / voxel_size
    bs = (box_dimensions/2 + xsection_offsets)  / voxel_size

    lines = [[bs[1], bs[0]], [bs[2], bs[0]], [bs[1], bs[2]]]
    cs_names = ['X-Y cross-section', 'X-Z cross-section', 'Y-Z cross-section' ] 

    c_EM = np.flip(np.round(cc_EM).astype(int), axis=0)  # swapped to match the EM array index locations
    s_EM = np.flip(np.round(cs_EM).astype(int), axis=0)
    ci_EM = np.flip(np.round(cc_EM - bd_EM / 2.0).astype(int), axis=0)
    ca_EM = np.flip(np.round(cc_EM + bd_EM / 2.0).astype(int), axis=0)

    dx = ca_EM-ci_EM
    yratios = [dx[1], dx[0]] if XZ_section else [dx[1]]
    xratios = [dx[2], dx[0]] if ZY_section else [dx[2]]
    xy_ratio = np.sum(yratios) / np.sum(xratios)
    hor_size = 5.0
    vert_size = hor_size * xy_ratio
  
    # Build images
    images = []
    for i, s in enumerate(tqdm(s_EM, desc = 'Loading EM Cross-sections Data')):
        ci_x = ci_EM.copy()
        ca_x = ca_EM.copy()
        ci_x[i] = s
        ca_x[i] = s+1
        #print(ci_x[0],ca_x[0], ci_x[1],ca_x[1], ci_x[2],ca_x[2])
        #EM_crop = deepcopy(np.squeeze(mrc_obj.data[ci_x[0]:ca_x[0], ci_x[1]:ca_x[1], ci_x[2]:ca_x[2]]))
        EM_crop = np.squeeze(mrc_obj.data[ci_x[0]:ca_x[0], ci_x[1]:ca_x[1], ci_x[2]:ca_x[2]])
        #print('EM Crop Base: ', EM_crop.base)
        ysz, xsz = np.shape(EM_crop)
        print(time.strftime('%Y/%m/%d  %H:%M:%S   ')+cs_names[i] + ' loaded, dimensions (pixels):', xsz, ysz)
        if i==2:
            EM_crop = np.transpose(EM_crop)
        images.append(EM_crop)
    mrc_obj.close()

    # Determine EM data range
    EM_mins = []
    EM_maxs =[]  
    for img in tqdm(images, desc='Determining EM data range'):
        ysz, xsz = np.shape(img)
        vmin, vmax = get_min_max_thresholds(img[ysz//5:ysz//5*4, xsz//5:xsz//5*4], thr_min = 1e-3, thr_max=1e-3, disp_res=False)
        EM_mins.append(vmin)
        EM_maxs.append(vmax)
    
    EM_min = kwargs.get('EM_min', np.min(np.array(EM_mins)))
    EM_max = kwargs.get('EM_max', np.max(np.array(EM_maxs)))
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will use EM-data range: {:.1f} - {:.1f}'.format(EM_min, EM_max))
    
    # ---------------------------------------------------------------------
    # now generate figures
    ncols = 2 if ZY_section else 1
    nrows = 2 if XZ_section else 1
    if XZ_section and ZY_section: gs_kw={"height_ratios" : yratios, "width_ratios" : xratios}
    if XZ_section and (not ZY_section): gs_kw={"height_ratios" : yratios}
    if (not XZ_section) and ZY_section: gs_kw={"width_ratios" : xratios}

    fig, axs = plt.subplots(nrows, ncols, figsize=(hor_size, vert_size), 
                            gridspec_kw=gs_kw, dpi=dpi)
    if XZ_section and ZY_section:
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.025*(xy_ratio**2)+addtl_sp, hspace=0.025)
    if XZ_section and (not ZY_section):
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, hspace=0.025)
    if (not XZ_section) and ZY_section:
        fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.025*(xy_ratio**2)+addtl_sp)
    
    print('Generating Cross-Section Images')
    
    if XZ_section and ZY_section:
        widths_um = [stack_size[0], stack_size[0], stack_size[2]]
        for x in np.arange(ncols):
            for y in np.arange(nrows):
                j=y*2+x
                if j<3:
                    if x==0 or (x==1 and XZ_section) or y==0 or (y==1 and ZY_section):
                        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Generating '+ cs_names[j])
                        axs[x,y].imshow(images[j], cmap='Greys', vmin=EM_min, vmax=EM_max, clip_on=True)
                        if XZ_section:
                            axs[x,y].axhline(lines[j][0], c = xsection_line_color, linewidth = xsection_linewidth)
                        if ZY_section:
                            axs[x,y].axvline(lines[j][1], c = xsection_line_color, linewidth = xsection_linewidth)
                        axs[x,y].set_xticks([])
                        axs[x,y].set_yticks([])
                        for sp in ['bottom','top', 'left', 'right']:
                            axs[x,y].spines[sp].set_color('white')
                            axs[x,y].spines[sp].set_linewidth(0.5)
                    if display_scale_bars:
                        if widths_um[j] > bar_length_um:
                            add_scale_bar(axs[x,y], pixel_size_um = voxel_size[j], **bar_kwargs)
        fig.delaxes(axs[1,1])
        
    if (XZ_section and (not ZY_section)) or (ZY_section and (not XZ_section)):
        for x in np.arange(2):
            j=x if XZ_section else 2*x
            axs[x].imshow(images[j], cmap='Greys', vmin=EM_min, vmax=EM_max, clip_on=True)
            if XZ_section:
                axs[x].axhline(lines[j][0], c = xsection_line_color, linewidth = xsection_linewidth)
                add_scale_bar(axs[x], pixel_size_um = voxel_size[2], **bar_kwargs)
            if ZY_section:
                axs[x].axvline(lines[j][1], c = xsection_line_color, linewidth = xsection_linewidth)
                add_scale_bar(axs[x], pixel_size_um = voxel_size[0], **bar_kwargs)
            axs[x].set_xticks([])
            axs[x].set_yticks([])
            for sp in ['bottom','top', 'left', 'right']:
                axs[x].spines[sp].set_color('white')
                axs[x].spines[sp].set_linewidth(0.5)

    if save_PNG:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving image into: ', save_filename)
        fig.savefig(save_filename, dpi=dpi, transparent=True)
        
    return images, axs


def bin_crop_frames(bin_crop_parameters):
    '''
    Read frames from MRC bin/crop them and return to be saved.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com
    
    Parameters:
    ---------
    params : list
        mrc_filename : filename for the MRC source file
        dtp : data type
        start_frame_ID : int
            start frame to read
        stop_frame_ID : int
            stop frame to read
        target_frame_ID : int
            frame to save
        xbin_factor, ybin_factor, zbin_factor,
        mode, flipY
        xi, xa, yi, ya
    Returns : target_frame_ID, transformed_frame
    '''
    mrc_filename, dtp, start_frame_ID, stop_frame_ID, target_frame_ID, xbin_factor, ybin_factor, zbin_factor, mode, flipY, xi, xa, yi, ya = bin_crop_parameters
    nx_binned = (xa-xi)//xbin_factor
    ny_binned = (ya-yi)//ybin_factor
    xa = xi + nx_binned * xbin_factor
    ya = yi + ny_binned * ybin_factor
    mrc_filename  = os.path.normpath(mrc_filename)
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    if mode == 'mean':
        zbinnd_fr = np.mean(mrc_obj.data[start_frame_ID:stop_frame_ID, yi:ya, xi:xa].astype(np.float32), axis=0)
    else:
        zbinnd_fr = np.sum(mrc_obj.data[start_frame_ID:stop_frame_ID, yi:ya, xi:xa].astype(np.float32), axis=0)
    if (xbin_factor > 1) or (ybin_factor > 1):
        if mode == 'mean':
            zbinnd_fr = np.mean(np.mean(zbinnd_fr.reshape(ny_binned, ybin_factor, nx_binned, xbin_factor), axis=3), axis=1)
        else:
            zbinnd_fr = np.sum(np.sum(zbinnd_fr.reshape(ny_binned, ybin_factor, nx_binned, xbin_factor), axis=3), axis=1)
    mrc_obj.close()
    if flipY:
        zbinnd_fr = np.flip(zbinnd_fr, axis = 0)
    return target_frame_ID, zbinnd_fr.astype(dtp)


def bin_crop_mrc_stack(mrc_filename, **kwargs):
    '''
    Bins and crops a 3D mrc stack along X-, Y-, or Z-directions and saves it into MRC or HDF5 format. ©G.Shtengel 08/2022 gleb.shtengel@gmail.com

    Parameters:
    ---------
        mrc_filename : str
            name (full path) of the mrc file to be binned.
    kwargs:
    ---------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3.
        max_futures : int
            max number of running futures. Default is 5000.
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is ['mrc']. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        zbin_factor : int
            binning factor in z-direction.
        xbin_factor : int
            binning factor in x-direction.
        ybin_factor : int
            binning factor in y-direction.
        mode  : str
            Binning mode. Default is 'mean', other option is 'sum'.
        mrc_mode : int
            mrc mode.
        flipY : boolean
            If Trye, the data will be flipped along Y axis (0 index) AFTER cropping.
        invert_data : boolean
            If True, invert the data.
        binned_copped_filename : str
            name (full path) of the mrc file to save the results into. If not present, the new file name is constructed from the original by adding "_zbinXX" at the end.
        xi : int
            left edge of the crop.
        xa : int
            right edge of the crop.
        yi : int
            top edge of the crop.
        ya : int
            bottom edge of the crop.
        fri : int
            start frame.
        fra : int
            stop frame.
        voxel_size_new : rec array.
            new voxel size in nm. Will be converted into Angstroms for MRC header.
    Returns:
        fnms_saved : list of str
            Names of the new (binned and cropped) data files.
    '''
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    max_futures = kwargs.get('max_futures', 5000)
    use_DASK, status_update_address = check_DASK(DASK_client)
    
    mrc_filename  = os.path.normpath(mrc_filename)

    fnm_types = kwargs.get("fnm_types", ['mrc'])
    xbin_factor = kwargs.get("xbin_factor", 1)      # binning factor in in x-direction
    ybin_factor = kwargs.get("ybin_factor", 1)      # binning factor in in y-direction
    zbin_factor = kwargs.get("zbin_factor", 1)      # binning factor in in z-direction

    mode = kwargs.get('mode', 'mean')                   # binning mode. Default is 'mean', other option is 'sum'
    flipY = kwargs.get('flipY', False)
    invert_data = kwargs.get('invert_data', False)
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    header = mrc_obj.header
    '''
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    '''
    mrc_mode =  kwargs.get('mrc_mode', mrc_obj.header.mode)
    dtp = np.int16
    if mrc_mode == 0:
        dtp = np.int8
    if mrc_mode == 2:
        dtp = np.float32
    if mrc_mode == 4:
        dtp = np.complex64
    if mrc_mode == 6:
        dtp = np.uint16

    voxel_size_angstr = mrc_obj.voxel_size
    voxel_size_angstr_new = voxel_size_angstr.copy()
    voxel_size_angstr_new.x = voxel_size_angstr.x * xbin_factor
    voxel_size_angstr_new.y = voxel_size_angstr.y * ybin_factor
    voxel_size_angstr_new.z = voxel_size_angstr.z * zbin_factor
    voxel_size_new = voxel_size_angstr.copy()
    voxel_size_new.x = voxel_size_angstr_new.x / 10.0
    voxel_size_new.y = voxel_size_angstr_new.y / 10.0
    voxel_size_new.z = voxel_size_angstr_new.z / 10.0
    try:
        voxel_size_new = kwargs.get('voxel_size_new', voxel_size_new)
        voxel_size_angstr_new.x = voxel_size_new.x * 10.0
        voxel_size_angstr_new.y = voxel_size_new.y * 10.0
        voxel_size_angstr_new.z = voxel_size_new.z * 10.0
    except:
        print('Incorrect voxel size entry')
        print('will use : ', voxel_size_new)
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    xi = kwargs.get('xi', 0)
    xa = kwargs.get('xa', nx)
    yi = kwargs.get('yi', 0)
    ya = kwargs.get('ya', ny)
    fri = kwargs.get('fri', 0)
    fra = kwargs.get('fra', nz)
    nx_binned = (xa-xi)//xbin_factor
    ny_binned = (ya-yi)//ybin_factor
    xa = xi + nx_binned * xbin_factor
    ya = yi + ny_binned * ybin_factor
    binned_copped_filename_default = os.path.splitext(mrc_filename)[0] + '_binned_croped.mrc'
    binned_copped_filename = kwargs.get('binned_copped_filename', binned_copped_filename_default)
    binned_mrc_filename = os.path.splitext(binned_copped_filename)[0] + '.mrc'
    binned_mrc_filename = os.path.normpath(binned_mrc_filename)
    dt = type(mrc_obj.data[0,0,0])
    print('Source mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
    print('Source Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
    if mode == 'sum':
        mrc_mode = 1
        dtp = np.int16
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Result mrc_mode: {:d}, source data type:'.format(mrc_mode), dtp)
    st_frames = np.arange(fri, fra, zbin_factor)
    mrc_obj.close()
    
    desc = 'Building Parameters Sets'
    params_mult = []
    for j, st_frame in enumerate(tqdm(st_frames, desc=desc)):
        params = [mrc_filename, dt, st_frame, (min(st_frame+zbin_factor, nz-1)), j, xbin_factor, ybin_factor, zbin_factor, mode, flipY, xi, xa, yi, ya]
        params_mult.append(params)
    
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   New Data Set Shape:  {:d} x {:d} x {:d}'.format(nx_binned, ny_binned, len(st_frames)))
    
    fnms_saved = []
    if 'mrc' in fnm_types:
        fnms_saved.append(binned_mrc_filename)
        mrc_new = mrcfile.new_mmap(binned_mrc_filename, shape=(len(st_frames), ny_binned, nx_binned), mrc_mode=mrc_mode, overwrite=True)
        mrc_new.voxel_size = voxel_size_angstr_new
        #mrc_new.header.cella = voxel_size_angstr_new
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr_new.x, voxel_size_angstr_new.y, voxel_size_angstr_new.z))
        desc = 'Saving the data stack into MRC file'

    if 'h5' in fnm_types:
        binned_h5_filename = os.path.splitext(binned_mrc_filename)[0] + '.h5'
        try:
            os.remove(binned_h5_filename)
        except:
            pass
        fnms_saved.append(binned_h5_filename)
        bdv_writer = npy2bdv.BdvWriter(binned_h5_filename, nchannels=1, blockdim=((1, 256, 256),))
        bdv_writer.append_view(stack=None, virtual_stack_dim=(len(st_frames),ny_binned,nx_binned),
                    time=0, channel=0,
                    voxel_size_xyz=(voxel_size_new.x, voxel_size_new.y, voxel_size_new.z), voxel_units='nm')
        if 'mrc' in fnm_types:
            desc = 'Saving the data stack into MRC and H5 files'
        else:
            desc = 'Saving the data stack into H5 file'
    
    if use_DASK:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
        #futures = DASK_client.map(bin_crop_frames, params_mult, retries = DASK_client_retries)
        # In case of a large source file, need to stadge the DASK jobs - cannot start all at once.
        DASK_batch = 0
        while len(params_mult) > max_futures:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs, {:d} jobs remaining'.format(DASK_batch, max_futures, (len(params_mult)-max_futures)))
            futures = [DASK_client.submit(bin_crop_frames, params) for params in params_mult[0:max_futures]]
            params_mult = params_mult[max_futures:]
            DASK_batch += 1
        
            for future in as_completed(futures):
                j, binned_cropped_fr = future.result()
                if 'mrc' in fnm_types:
                    if invert_data:
                        if mrc_mode == 0:  # uint8
                            binned_cropped_fr = 255 - binned_cropped_fr
                        if mrc_mode == 6:  # uint16
                            binned_cropped_fr = 65535 - binned_cropped_fr
                        if mrc_mode != 0 and mrc_mode != 6:
                            binned_cropped_fr = np.invert(binned_cropped_fr)
                    mrc_new.data[j,:,:] = binned_cropped_fr.astype(dtp)
                if 'h5' in fnm_types:
                    bdv_writer.append_plane(plane=binned_cropped_fr.astype(dtp), z=j, time=0, channel=0)
                future.cancel()

        if len(params_mult) > 0:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs'.format(DASK_batch, len(params_mult)))
            futures = [DASK_client.submit(bin_crop_frames, params) for params in params_mult]
            for future in as_completed(futures):
                j, binned_cropped_fr = future.result()
                if 'mrc' in fnm_types:
                    if invert_data:
                        if mrc_mode == 0:  # uint8
                            binned_cropped_fr = 255 - binned_cropped_fr
                        if mrc_mode == 6:  # uint16
                            binned_cropped_fr = 65535 - binned_cropped_fr
                        if mrc_mode != 0 and mrc_mode != 6:
                            binned_cropped_fr = np.invert(binned_cropped_fr)
                    mrc_new.data[j,:,:] = binned_cropped_fr.astype(dtp)
                if 'h5' in fnm_types:
                    bdv_writer.append_plane(plane=binned_cropped_fr.astype(dtp), z=j, time=0, channel=0)
                future.cancel()

    else:
        desc = 'Performing local computations'
        for params in tqdm(params_mult, desc = desc):
            j, binned_cropped_fr = bin_crop_frames(params)
            if 'mrc' in fnm_types:
                if invert_data:
                    if mrc_mode == 0:  # uint8
                        binned_cropped_fr = 255 - binned_cropped_fr
                    if mrc_mode == 6:  # uint16
                        binned_cropped_fr = 65535 - binned_cropped_fr
                    if mrc_mode != 0 and mrc_mode != 6:
                        binned_cropped_fr = np.invert(binned_cropped_fr)
                mrc_new.data[j,:,:] = binned_cropped_fr.astype(dtp)
            if 'h5' in fnm_types:
                bdv_writer.append_plane(plane=binned_cropped_fr.astype(dtp), z=j, time=0, channel=0)

    if 'mrc' in fnm_types:
        mrc_new.close()

    if 'h5' in fnm_types:
        bdv_writer.write_xml()
        bdv_writer.close()

    return fnms_saved


def bin_crop_tiff_files(params):
    '''
    Bins tiff files. This routine is used by merge_tiff_files_mrc_stack. ©G.Shtengel 05/2025 gleb.shtengel@gmail.com

    Parameters:
        params : list of parameters
        params = mrc_filename, dtp, fls_tiff, j, xbin_factor, ybin_factor, zbin_factor, mode, flipY, xi, xa, yi, ya

    Returns:
        j, frame
    '''
    mrc_filename, dtp, fls_tiff, j, xbin_factor, ybin_factor, mode, flipY, xi, xa, yi, ya = params
    nx_binned = (xa-xi)//xbin_factor
    ny_binned = (ya-yi)//ybin_factor
    xa = xi + nx_binned * xbin_factor
    ya = yi + ny_binned * ybin_factor
    frame = np.zeros((ny_binned, nx_binned), dtype=float)
    for fl_tiff in fls_tiff:
        init_frame = tiff.imread(fl_tiff)[yi:ya, xi:xa]
        binned_frame = np.mean(np.mean(init_frame.reshape(ny_binned, ybin_factor, nx_binned, xbin_factor), axis=3), axis=1)
        frame = frame + binned_frame
    return j, frame//len(fls_tiff)


def merge_tiff_files_mrc_stack(fls_tiff, **kwargs):
    '''
    Bins and crops a stack tiff files (frames) along X-, Y-, or Z-directions and saves them into MRC or HDF5 file. ©G.Shtengel 05/2025 gleb.shtengel@gmail.com

    Parameters:
        fls_tiff : list of str
            list of tiff file names (full paths) to be merged
    **kwargs:
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3.
        max_futures : int
            max number of running futures. Default is 5000.
        fnm_types : list of strings.
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is ['mrc']. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        zbin_factor : int
            binning factor in z-direction
        xbin_factor : int
            binning factor in x-direction
        ybin_factor : int
            binning factor in y-direction
        mode  : str
            Binning mode. Default is 'mean', other option is 'sum'
        mrc_mode : int
            mrc mode
        flipY : boolean
            If Trye, the data will be flipped along Y axis (0 index) AFTER cropping.
        invert_data : boolean
            If True, invert the data
        mrc_filename : str
            name (full path) of the mrc file to save the results into. If not present, the new file name is constructed from the original by adding "_merged.mrc" at the end.
        xi : int
            left edge of the crop
        xa : int
            right edge of the crop
        yi : int
            top edge of the crop
        ya : int
            bottom edge of the crop
        fri : int
            start frame
        fra : int
            stop frame
        voxel_size_angstr : rec array
            voxel size in Angstroms. Default is 80.0 x 80.0 x 80.0.
    Returns:
        fnms_saved : list of str
            Names of the new (binned and cropped) data files.
    '''
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    max_futures = kwargs.get('max_futures', 5000)
    use_DASK, status_update_address = check_DASK(DASK_client)
    mrc_filename_default = os.path.splitext(fls_tiff[0])[0] + '_merged.mrc'
    mrc_filename = kwargs.get('mrc_filename', mrc_filename_default)
    mrc_filename  = os.path.normpath(mrc_filename)

    fnm_types = kwargs.get("fnm_types", ['mrc'])
    xbin_factor = kwargs.get("xbin_factor", 1)      # binning factor in in x-direction
    ybin_factor = kwargs.get("ybin_factor", 1)      # binning factor in in y-direction
    zbin_factor = kwargs.get("zbin_factor", 1)      # binning factor in in z-direction

    mode = kwargs.get('mode', 'mean')                   # binning mode. Default is 'mean', other option is 'sum'
    flipY = kwargs.get('flipY', False)
    invert_data = kwargs.get('invert_data', False)

    mrc_mode = kwargs.get('mrc_mode', 1)
    
    if mode == 'sum' and mrc_mode == 0:
        mrc_mode = 1

    if mrc_mode == 0:
        dtp = np.int8
    if mrc_mode == 1:
        dtp = np.int16
    if mrc_mode == 2:
        dtp = np.float32
    if mrc_mode == 4:
        dtp = np.complex64
    if mrc_mode == 6:
        dtp = np.uint16

    voxel_size_angstr_default = np.rec.fromrecords(80, names=['x', 'y', 'z'], formats=[np.float32, np.float32, np.float32])
    voxel_size_angstr = kwargs.get('voxel_size_angstr', voxel_size_angstr_default)

    test_tif = tiff.imread(fls_tiff[0])
    ny, nx = test_tif.shape
    nz = len(fls_tiff)
    xi = kwargs.get('xi', 0)
    xa = kwargs.get('xa', nx)
    yi = kwargs.get('yi', 0)
    ya = kwargs.get('ya', ny)
    fri = kwargs.get('fri', 0)
    fra = kwargs.get('fra', nz)
    nx_binned = (xa-xi)//xbin_factor
    ny_binned = (ya-yi)//ybin_factor
    xa = xi + nx_binned * xbin_factor
    ya = yi + ny_binned * ybin_factor

    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Result mrc_mode: {:d}, source data type:'.format(mrc_mode), dtp)
    st_frames = np.arange(fri, fra, zbin_factor)
    
    desc = 'Building Parameters Sets'
    params_mult = []
    for j, st_frame in enumerate(tqdm(st_frames, desc=desc)):
        params = [mrc_filename, dtp, fls_tiff[st_frame : (min(st_frame+zbin_factor, nz-1))], j, xbin_factor, ybin_factor, mode, flipY, xi, xa, yi, ya]
        params_mult.append(params)
    
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   New Data Set Shape:  {:d} x {:d} x {:d}'.format(nx_binned, ny_binned, len(st_frames)))
    
    fnms_saved = []
    if 'mrc' in fnm_types:
        fnms_saved.append(mrc_filename)
        mrc_new = mrcfile.new_mmap(mrc_filename, shape=(len(st_frames), ny_binned, nx_binned), mrc_mode=mrc_mode, overwrite=True)
        mrc_new.voxel_size = voxel_size_angstr
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
        desc = 'Saving the data stack into MRC file'

    if 'h5' in fnm_types:
        binned_h5_filename = os.path.splitext(mrc_filename)[0] + '.h5'
        try:
            os.remove(binned_h5_filename)
        except:
            pass
        fnms_saved.append(binned_h5_filename)
        bdv_writer = npy2bdv.BdvWriter(binned_h5_filename, nchannels=1, blockdim=((1, 256, 256),))
        bdv_writer.append_view(stack=None, virtual_stack_dim=(len(st_frames),ny_binned,nx_binned),
                    time=0, channel=0,
                    voxel_size_xyz=(voxel_size_new.x, voxel_size_new.y, voxel_size_new.z), voxel_units='nm')
        if 'mrc' in fnm_types:
            desc = 'Saving the data stack into MRC and H5 files'
        else:
            desc = 'Saving the data stack into H5 file'
    
    if use_DASK:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
        #futures = DASK_client.map(bin_crop_frames, params_mult, retries = DASK_client_retries)
        # In case of a large source file, need to stadge the DASK jobs - cannot start all at once.
        DASK_batch = 0
        while len(params_mult) > max_futures:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs, {:d} jobs remaining'.format(DASK_batch, max_futures, (len(params_mult)-max_futures)))
            futures = [DASK_client.submit(bin_crop_tiff_files, params) for params in params_mult[0:max_futures]]
            params_mult = params_mult[max_futures:]
            DASK_batch += 1
            
            for future in as_completed(futures):
                j, binned_cropped_fr = future.result()
                if 'mrc' in fnm_types:
                    if invert_data:
                        if mrc_mode == 0:  # uint8
                            binned_cropped_fr = 255 - binned_cropped_fr
                        if mrc_mode == 6:  # uint16
                            binned_cropped_fr = 65535 - binned_cropped_fr
                        if mrc_mode != 0 and mrc_mode != 6:
                            binned_cropped_fr = np.invert(binned_cropped_fr)
                    mrc_new.data[j,:,:] = binned_cropped_fr.astype(dtp)
                if 'h5' in fnm_types:
                    bdv_writer.append_plane(plane=binned_cropped_fr.astype(dtp), z=j, time=0, channel=0)
                future.cancel()
                
        if len(params_mult) > 0:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs'.format(DASK_batch, len(params_mult)))
            futures = [DASK_client.submit(bin_crop_tiff_files, params) for params in params_mult]
            for future in as_completed(futures):
                j, binned_cropped_fr = future.result()
                if 'mrc' in fnm_types:
                    if invert_data:
                        if mrc_mode == 0:  # uint8
                            binned_cropped_fr = 255 - binned_cropped_fr
                        if mrc_mode == 6:  # uint16
                            binned_cropped_fr = 65535 - binned_cropped_fr
                        if mrc_mode != 0 and mrc_mode != 6:
                            binned_cropped_fr = np.invert(binned_cropped_fr)
                    mrc_new.data[j,:,:] = binned_cropped_fr.astype(dtp)
                if 'h5' in fnm_types:
                    bdv_writer.append_plane(plane=binned_cropped_fr.astype(dtp), z=j, time=0, channel=0)
                future.cancel()


    else:
        desc = 'Performing local computations'
        for params in tqdm(params_mult, desc = desc):
            j, binned_cropped_fr = bin_crop_tiff_files(params)
            if 'mrc' in fnm_types:
                if invert_data:
                    if mrc_mode == 0:  # uint8
                        binned_cropped_fr = 255 - binned_cropped_fr
                    if mrc_mode == 6:  # uint16
                        binned_cropped_fr = 65535 - binned_cropped_fr
                    if mrc_mode != 0 and mrc_mode != 6:
                        binned_cropped_fr = np.invert(binned_cropped_fr)
                mrc_new.data[j,:,:] = binned_cropped_fr.astype(dtp)
            if 'h5' in fnm_types:
                bdv_writer.append_plane(plane=binned_cropped_fr.astype(dtp), z=j, time=0, channel=0)

    if 'mrc' in fnm_types:
        mrc_new.close()

    if 'h5' in fnm_types:
        bdv_writer.write_xml()
        bdv_writer.close()

    return fnms_saved


def destreak_single_frame_kernel_shared(destreak_kernel, params):
    '''
    Read a single frame from MRC stack, destreak the data by performing FFT, multiplying it by kernel, and performing inverse FFT.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com
    
    Parameters:
    destreak_kernel : 2D array - to multiply the FFT
    params : list
        mrc_filename : filename for the MRC source file
        source_frame_ID : int
            frame to read / destreak
        target_frame_ID : int
            frame to save
        data_min, data_max : floats
    Returns : target_frame_ID, transformed_frame
    '''
    mrc_filename, dt, source_frame_ID, target_frame_ID, data_min, data_max, partial_destreaking_params = params
    partial_destreaking, transition_direction, flip_transitionX, flip_transitionY, xi, xa, yi, ya = partial_destreaking_params
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    read_fr = mrc_obj.data[source_frame_ID, :, :]
    mrc_obj.close()

    padded_fr, clip_mask = clip_pad_image(read_fr, data_min, data_max)
    destreaked_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(padded_fr))) * destreak_kernel
    transformed_frame = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(destreaked_fft)))).astype(float)
    if partial_destreaking:
        transformed_frame = merge_images_with_transition(padded_fr, transformed_frame, transition_direction=transition_direction, flip_transitionX=flip_transitionX, flip_transitionY=flip_transitionY, xi=xi, xa=xa, yi=yi, ya=ya) * clip_mask
    else:
        transformed_frame = transformed_frame * clip_mask
    
    return target_frame_ID, transformed_frame


def destreak_mrc_stack_with_kernel(mrc_filename, destreak_kernel, data_min, data_max, **kwargs):
    '''
    Read MRC stack, destreak the data by performing FFT, multiplying it by kernel, and performing inverse FFT, and save it into MRC or H5 stack.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com
    Use one of the following to calculate the de-streaking kernel:
        build_kernel_FFT_zero_destreaker_radii_angles(data, **kwargs):
        build_kernel_FFT_zero_destreaker_XY(data, **kwargs):
        build_kernel_FFT_destreaker_autodetect(data, **kwargs):

    Parameters:
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
    destreak_kernel : 2D array - to multiply the FFT
    data_min, data_max : floats
        In case if the data has been padded by constant values at the edges during the previous registrations steps,
        it will need to be replaced temporarily by a fake "real" data - otherwise FFT will be distorted.
        The padded values (that should out of range data_min-data_max) will be:
         - identified by comparing to data_min and data_max
         - if ouside the above range - replaced by mirror imaged adjacent data
         - after FFT, kernel multiplication, and reverse FFT, they will be replaced by zeros.
    kwargs:
    ---------
    DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    max_futures : int
        max number of running futures. Default is 5000.
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames.
    invert_data : boolean
        If True, the data will be inverted.
    fri : int
        start frame
    fra : int
        stop frame
    voxel_size : rec array of 3 elements
        voxel size in nm. Default is the data that was stored in the initial file.
    disp_res : boolean
        Display messages and intermediate results
    fnm_types : list of strings
        File type(s) for output data. Options are: ['h5', 'mrc'].
        Defauls is 'mrc'. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
    partial_destreaking : boolean
        Default is False. if True, only part of the image is destreaked. This is accomplished by destreaking the full image and then building a composite image with a smooth transition from non-destreaked to destreaked.
    transition_direction : str
        'Y' (default), or 'X'.
    flip_transitionY : boolean
        Default is False. flip transition in Y direction.
    flip_transitionX : boolean
        Default is False. flip transition in X direction.
    xi : int
        Start index of transition if transition_direction is 'X'. Default is 1/2 of the image.
    xa : int
        Stop index of transition if transition_direction is 'X'. Default is 3/4 of the image.
    yi : int
        Start index of transition if transition_direction is 'Y'. Default is 1/2 of the image.
    ya : int
        Stop index of transition if transition_direction is 'Y'. Default is 3/4 of the image.
    save_filename : str
        Path to the filename to save the results. If empty, mrc_filename+'_destreaked.mrc' will be used.
    
    Returns:
        fnms_saved : list of str    Names of the new (binned and cropped) data files.
    '''
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    max_futures = kwargs.get('max_futures', 5000)
    disp_res  = kwargs.get("disp_res", False )
    fnm_types = kwargs.get("fnm_types", ['mrc'])
    use_DASK, status_update_address = check_DASK(DASK_client)
    
    mrc_filename  = os.path.normpath(mrc_filename)
    save_filename = kwargs.get('save_filename', mrc_filename.replace('.mrc', '_destreaked.mrc'))

    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    header = mrc_obj.header
    '''
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    '''
    mrc_mode = mrc_obj.header.mode
    voxel_size_angstr_from_mrc = mrc_obj.voxel_size
    voxel_size_from_mrc = voxel_size_angstr_from_mrc.copy()
    voxel_size_from_mrc.x = voxel_size_angstr_from_mrc.x/1000.0
    voxel_size_from_mrc.y = voxel_size_angstr_from_mrc.y/1000.0
    voxel_size_from_mrc.z = voxel_size_angstr_from_mrc.z/1000.0
    voxel_size = kwargs.get("voxel_size", voxel_size_from_mrc)
    voxel_size_angstr = voxel_size.copy()
    voxel_size_angstr.x = voxel_size.x*1000.0
    voxel_size_angstr.y = voxel_size.y*1000.0
    voxel_size_angstr.z = voxel_size.z*1000.0
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    fri = kwargs.get('fri', 0)
    fra = kwargs.get('fra', nz)
    st_frames = np.arange(fri, fra)
    nz_new = len(st_frames)

    partial_destreaking = kwargs.get('partial_destreaking', False)
    transition_direction = kwargs.get('transition_direction', 'Y')
    flip_transitionY = kwargs.get('flip_transitionY', False)
    flip_transitionX = kwargs.get('flip_transitionX', False)
    xi = kwargs.get('xi', nx//2)
    xa = kwargs.get('xa', nx//4*3)
    yi = kwargs.get('yi', ny//2)
    ya = kwargs.get('ya', ny//4*3)
    partial_destreaking_params = [partial_destreaking, transition_direction, flip_transitionX, flip_transitionY, xi, xa, yi, ya]

    dt = type(mrc_obj.data[0,0,0])
    mrc_obj.close()
    
    if disp_res:
        print('Source mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
        print('Source Data Shape:  {:d} x {:d} x {:d}'.format(nx, ny, nz))
        print('Source Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr_from_mrc.x, voxel_size_angstr_from_mrc.y, voxel_size_angstr_from_mrc.z))
    mrc_mode = 1
    dt = np.int16
    if disp_res:
        print('Result mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
        print('Result Data Shape:  {:d} x {:d} x {:d}'.format(nx, ny, nz_new))
        print('Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
    
    fnms_saved = []
    if 'mrc' in fnm_types:
        mrc_new = mrcfile.new_mmap(save_filename, shape=(nz_new, ny, nx), mrc_mode=mrc_mode, overwrite=True)
        mrc_new.voxel_size = voxel_size_angstr

    if 'h5' in fnm_types:
        save_filename_h5 = save_filename.replace('.mrc', '.h5')
        try:
            os.remove(save_filename_h5)
        except:
            pass
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving dataset into Big Data Viewer HDF5 file: ', save_filename_h5)
        bdv_writer = npy2bdv.BdvWriter(save_filename_h5, nchannels=1, blockdim=((1, 256, 256),))
        bdv_writer.append_view(stack=None,
            virtual_stack_dim=(nz_new, ny, nx),
            time=0,
            channel=0,
            voxel_size_xyz=(voxel_size.x, voxel_size.y, voxel_size.z),
            voxel_units='nm')                   
    
    desc = 'Creating params list'
    params_mult = []
    for j, st_frame in enumerate(tqdm(st_frames, desc=desc)):
        params = [mrc_filename, dt, st_frame, j, data_min, data_max, partial_destreaking_params]
        params_mult.append(params)
    
    if use_DASK:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
        [destreak_kernel_future] = DASK_client.scatter([destreak_kernel], broadcast=True)
        #futures = DASK_client.map(bin_crop_frames, params_mult, retries = DASK_client_retries)
        # In case of a large source file, need to stadge the DASK jobs - cannot start all at once.
        DASK_batch = 0
        while len(params_mult) > max_futures:
            DASK_batch += 1
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs, {:d} jobs remaining'.format(DASK_batch, max_futures, (len(params_mult)-max_futures)))
            futures = [DASK_client.submit(destreak_single_frame_kernel_shared, destreak_kernel_future, params) for params in params_mult[0:max_futures]]
            params_mult = params_mult[max_futures:]
            for future in as_completed(futures):
                target_frame_ID, transformed_frame = future.result()
                if 'mrc' in fnm_types:
                    mrc_new.data[target_frame_ID,:,:] = transformed_frame
                if 'h5' in fnm_types:
                    bdv_writer.append_plane(plane=transformed_frame, z=target_frame_ID, time=0, channel=0)
                future.cancel()
        if len(params_mult) > 0:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs'.format(DASK_batch, len(params_mult)))
            futures = [DASK_client.submit(destreak_single_frame_kernel_shared, destreak_kernel_future, params) for params in params_mult]
            for future in as_completed(futures):
                target_frame_ID, transformed_frame = future.result()
                if 'mrc' in fnm_types:
                    mrc_new.data[target_frame_ID,:,:] = transformed_frame
                if 'h5' in fnm_types:
                    bdv_writer.append_plane(plane=transformed_frame, z=target_frame_ID, time=0, channel=0)
                future.cancel()
        '''   this is what processing used to be before I added staging in case of a very large data set
        [future] = DASK_client.scatter([destreak_kernel], broadcast=True)
        futures = [DASK_client.submit(destreak_single_frame_kernel_shared, future, params) for params in params_mult]
        #futures = DASK_client.map(destreak_single_frame_kernel_shared, params_mult, retries = DASK_client_retries)       
        for future in as_completed(futures):
            target_frame_ID, transformed_frame = future.result()
            mrc_new.data[target_frame_ID,:,:] = transformed_frame
            future.cancel()
        '''
    else:
        desc = 'Saving the destreaked data stack into MRC file'    
        for params in tqdm(params_mult, desc=desc):
            target_frame_ID, transformed_frame = destreak_single_frame_kernel_shared(destreak_kernel, params)
            if 'mrc' in fnm_types:
                mrc_new.data[target_frame_ID,:,:] = transformed_frame
            if 'h5' in fnm_types:
                bdv_writer.append_plane(plane=transformed_frame, z=target_frame_ID, time=0, channel=0)

    if 'mrc' in fnm_types:
        mrc_new.close()
        fnms_saved.append(save_filename)
    if 'h5' in fnm_types:
        bdv_writer.write_xml()
        bdv_writer.close()
        fnms_saved.append(save_filename_h5)

    return fnms_saved


def smooth_single_frame_kernel_shared(smooth_kernel, params):
    '''
    Read a single frame from MRC stack, smooth the data by performing 2D-convolution with kernel.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com
    
    Parameters:
    ---------
    smooth_kernel : 2D array - for 2D-convolution
    params : list
        mrc_filename : filename for the MRC source file
        source_frame_ID : int
            frame to read / smooth
        target_frame_ID : int
            frame to save
        data_min, data_max : floats

    Returns : target_frame_ID, transformed_frame
    '''
    mrc_filename, dt, source_frame_ID, target_frame_ID, data_min, data_max = params
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    read_fr = mrc_obj.data[source_frame_ID, :, :]
    mrc_obj.close()
    padded_fr, clip_mask = clip_pad_image(read_fr, data_min, data_max)
    transformed_frame = convolve2d(padded_fr, smooth_kernel, mode='same').astype(float) * clip_mask
    
    return target_frame_ID, transformed_frame


def smooth_mrc_stack_with_kernel(mrc_filename, smooth_kernel, data_min, data_max, **kwargs):
    '''
    Read MRC stack, smooth the data by performing 2D-convolution with smooth_kernel, and save the data.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com

    Parameters:
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
    smooth_kernel : 2D array - for 2D-convolution
    data_min, data_max : floats
        In case if the data has been padded by constant values at the edges during the previous registrations steps,
        it will need to be replaced temporarily by a fake "real" data - otherwise FFT will be distorted.
        The padded values (that should out of range data_min-data_max) will be:
         - identified by comparing to data_min and data_max
         - if ouside the above range - replaced by mirror imaged adjacent data
         - after FFT, kernel multiplication, and reverse FFT, they will be replaced by zeros.
    kwargs:
    ---------
    DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    max_futures : int
        max number of running futures. Default is 5000.
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames.
    invert_data : boolean
        If True, the data will be inverted.
    fri : int
        start frame
    fra : int
        stop frame
    save_filename : str
        Path to the filename to save the results. If empty, mrc_filename+'_smoothed.mrc' will be used.

    Returns:
        save_filename : str
            The name of the smoothed MRC stack.
    '''
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    max_futures = kwargs.get('max_futures', 5000)
    use_DASK, status_update_address = check_DASK(DASK_client)
    
    mrc_filename  = os.path.normpath(mrc_filename)
    save_filename = kwargs.get('save_filename', mrc_filename.replace('.mrc', '_smoothed.mrc'))

    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    header = mrc_obj.header
    '''
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    '''
    mrc_mode = mrc_obj.header.mode
    voxel_size_angstr = mrc_obj.voxel_size
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    fri = kwargs.get('fri', 0)
    fra = kwargs.get('fra', nz)

    dt = type(mrc_obj.data[0,0,0])
    mrc_obj.close()
    
    print('Source mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
    print('Source Data Shape:  {:d} x {:d} x {:d}'.format(nx, ny, nz))
    print('Source Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
    mrc_mode = 1
    dt = np.int16
    print('Result mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
    st_frames = np.arange(fri, fra)
    print('New Data Shape:  {:d} x {:d} x {:d}'.format(nx, ny, len(st_frames)))
    
    mrc_new = mrcfile.new_mmap(save_filename, shape=(len(st_frames), ny, nx), mrc_mode=mrc_mode, overwrite=True)
    mrc_new.voxel_size = voxel_size_angstr
    #mrc_new.header.cella = voxel_size_angstr
    print('Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
    
    desc = 'Creating params list'
    # mrc_source_obj, mrc_target_obj, dt, source_frame, target_frame, smooth_kernel, data_min, data_max = params
    params_mult = []
    for j, st_frame in enumerate(tqdm(st_frames, desc=desc)):
        params = [mrc_filename, dt, st_frame, j, data_min, data_max]
        params_mult.append(params)

    if use_DASK:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
        [smooth_kernel_future] = DASK_client.scatter([smooth_kernel], broadcast=True)
        #futures = DASK_client.map(bin_crop_frames, params_mult, retries = DASK_client_retries)
        # In case of a large source file, need to stadge the DASK jobs - cannot start all at once.
        DASK_batch = 0
        while len(params_mult) > max_futures:
            DASK_batch += 1
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs, {:d} jobs remaining'.format(DASK_batch, max_futures, (len(params_mult)-max_futures)))
            futures = [DASK_client.submit(smooth_single_frame_kernel_shared, smooth_kernel_future, params) for params in params_mult[0:max_futures]]
            params_mult = params_mult[max_futures:]
            for future in as_completed(futures):
                target_frame_ID, transformed_frame = future.result()
                mrc_new.data[target_frame_ID,:,:] = transformed_frame
                future.cancel()
        if len(params_mult) > 0:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting DASK batch {:d} with {:d} jobs'.format(DASK_batch, len(params_mult)))
            futures = [DASK_client.submit(smooth_single_frame_kernel_shared, smooth_kernel_future, params) for params in params_mult]
            for future in as_completed(futures):
                target_frame_ID, transformed_frame = future.result()
                mrc_new.data[target_frame_ID,:,:] = transformed_frame
                future.cancel()
        ''' this is what it used to be before DASK future staging
        [future] = DASK_client.scatter([smooth_kernel], broadcast=True)
        futures = [DASK_client.submit(smooth_single_frame_kernel_shared, future, params) for params in params_mult]
        #futures = DASK_client.map(smooth_single_frame_kernel_shared, params_mult, retries = DASK_client_retries)       
        for future in as_completed(futures):
            target_frame_ID, transformed_frame = future.result()
            mrc_new.data[target_frame_ID,:,:] = transformed_frame
            future.cancel()
        '''
    else:
        desc = 'Saving the smoothed data stack into MRC file'    
        for params in tqdm(params_mult, desc=desc):
            target_frame_ID, transformed_frame = smooth_single_frame_kernel_shared(smooth_kernel, params)
            mrc_new.data[target_frame_ID,:,:] = transformed_frame
    mrc_new.close()
    
    return save_filename


def destreak_smooth_mrc_stack_with_kernels(mrc_filename, destreak_kernel, smooth_kernel, data_min, data_max, **kwargs):
    '''
    Read MRC stack and destreak the data by performing FFT, multiplying it by kernel, and performing inverse FFT.
    ©G.Shtengel, 10/2023. gleb.shtengel@gmail.com

    Parameters
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
    destreak_kernel : 2D array - to multiply the FFT
    smooth_kernel : 2D array - for 2D-convolution
    data_min, data_max : floats
        In case if the data has been padded by constant values at the edges during the previous registrations steps,
        it will need to be replaced temporarily by a fake "real" data - otherwise FFT and smoothing will be distorted.
        The padded values (that should out of range data_min-data_max) will be:
         - identified by comparing to data_min and data_max
         - if ouside the above range - replaced by mirror imaged adjacent data
         - after destreaking and smoothing they will be replaced by zeros.

    kwargs:
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames
    invert_data : boolean
        If True, the data will be inverted
    save_destreak_filename : str
        Path to the filename to save the destreaked stack. If empty, mrc_filename+'_destreaked.mrc' will be used
    save_destreak_smooth_filename : str
        Path to the filename to save the destreaked stack. If empty, mrc_filename+'_destreaked_smoothed.mrc' will be used
    
    Returns the name of the destreaked MRC stack
    '''
    mrc_filename  = os.path.normpath(mrc_filename)
    save_destreak_filename = kwargs.get('save_destreak_filename', mrc_filename.replace('.mrc', '_destreaked.mrc'))
    save_destreak_smooth_filename = kwargs.get('save_destreak_smooth_filename', mrc_filename.replace('.mrc', '_destreaked_smoothed.mrc'))

    mrc_obj = mrcfile.mmap(mrc_filename, mode='r', permissive=True)
    header = mrc_obj.header
    '''
        mode 0 -> uint8
        mode 1 -> int16
        mode 2 -> float32
        mode 4 -> complex64
        mode 6 -> uint16
    '''
    mrc_mode = mrc_obj.header.mode
    voxel_size_angstr = mrc_obj.voxel_size
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    fri = kwargs.get('fri', 0)
    fra = kwargs.get('fra', nz)

    dt = type(mrc_obj.data[0,0,0])
    print('Source mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
    print('Source Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
    mrc_mode = 1
    dt = np.int16
    print('Result mrc_mode: {:d}, source data type:'.format(mrc_mode), dt)
    st_frames = np.arange(fri, fra)
    print('New Data Set Shape:  {:d} x {:d} x {:d}'.format(nx, ny, len(st_frames)))

    mrc_new_destreaked = mrcfile.new_mmap(save_destreak_filename, shape=(len(st_frames), ny, nx), mrc_mode=mrc_mode, overwrite=True)
    mrc_new_destreaked.voxel_size = voxel_size_angstr
    mrc_new_destreaked_smoothed = mrcfile.new_mmap(save_destreak_smooth_filename, shape=(len(st_frames), ny, nx), mrc_mode=mrc_mode, overwrite=True)
    mrc_new_destreaked_smoothed.voxel_size = voxel_size_angstr
    #mrc_new_destreaked.header.cella = voxel_size_angstr
    #mrc_new_destreaked_smoothed.header.cella = voxel_size_angstr
    print('Result Voxel Size (Angstroms): {:2f} x {:2f} x {:2f}'.format(voxel_size_angstr.x, voxel_size_angstr.y, voxel_size_angstr.z))
    desc = 'Saving the destreaked data stacks into MRC files'

    for j, st_frame in enumerate(tqdm(st_frames, desc=desc)):
        read_fr = mrc_obj.data[st_frame, :, :]
        padded_fr, clip_mask = clip_pad_image(read_fr, data_min, data_max)
        destreaked_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(padded_fr))) * destreak_kernel
        destreaked_data = np.real(np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(destreaked_fft)))).astype(float)
        mrc_new_destreaked.data[j,:,:] = destreaked_data * clip_mask       
        mrc_new_destreaked_smoothed.data[j,:,:] = convolve2d(destreaked_data, smooth_kernel, mode='same').astype(float) * clip_mask
        
    mrc_new_destreaked.close()
    mrc_new_destreaked_smoothed.close()
    mrc_obj.close()

    return save_destreak_filename, save_destreak_smooth_filename


def mrc_stack_estimate_resolution_blobs_2D(mrc_filename, **kwargs):
    '''
    Estimate transitions in the images inside mrc_stack, uses select_blobs_LoG_analyze_transitions(frame_eval, **kwargs). gleb.shtengel@gmail.com  06/2023 
    Parameters:
    ---------
    mrc_filename : str
        File name (full path) of the mrc stack to be analyzed
        
    kwargs:
    ---------
    DASK_client : DASK client. If set to empty string '' (default), local computations are performed.
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    frame_inds : list of int
        List of frame indices to use to display the evaluation box.
        Default are [nfrs//10, nfrs//2, nfrs//10*9]
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
        If evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        If True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    data_dir : str
        data directory (path)
    Sample_ID : str
        Sample ID
    invert_data : boolean
        If True - the data is inverted
    flipY : boolean
        If True, the data will be flipped along Y-axis. Default is False.
    zbin_factor : int
    min_sigma : float
        Min sigma (in pixel units) for Gaussian kernel in LoG search. Default is 1.0
    max_sigma : float
        Max sigma (in pixel units) for Gaussian kernel in LoG search. Default is 1.5
    threshold : float
        Threshold for LoG search. Default is 0.005. The absolute lower bound for scale space maxima. Local maxima smaller
        than threshold are ignored. Reduce this to detect blobs with less intensities. 
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
        Error flag is incremented by 4 if the determined transition distance is below this value. Default is 0.0
    transition_high_limit : float
        Error flag is incremented by 8 if the determined transition distance is above this value. Default is 10.0
    verbose : boolean
        Print the outputs. Default is False.
    disp_res : boolean
        Display results. Default is False.
    title : str
        Title.
    nbins : int
        Number of bins for histogram.
    save_data_xlsx : boolean
        save the data into Excel workbook. Default is True.
    results_file_xlsx : file name for Excel workbook to save the results.

    Returns: XY_transitions ; array[len(frame_inds), 4]
        array consists of lines - one line for each frame in frame_inds
        each line has 4 elements: [Xpt1, Xpt2, Ypt1, Ypt2]
    '''
    kwargs['mrc_filename'] = mrc_filename
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    use_DASK, status_update_address = check_DASK(DASK_client)

    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])

    data_dir = kwargs.get("data_dir", os.path.dirname(mrc_filename))
    Sample_ID = kwargs.get("Sample_ID", '')

    invert_data = kwargs.get("invert_data", False)
    kwargs['invert_data'] = invert_data
    flipY = kwargs.get("flipY", False)
    kwargs['flipY'] = flipY
    zbin_factor = kwargs.get("zbin_factor", 1)
    kwargs['zbin_factor'] = zbin_factor
    min_sigma = kwargs.get('min_sigma', 1.0)
    max_sigma = kwargs.get('max_sigma', 1.0)

    overlap = kwargs.get('overlap', 0.1)
    subset_size = kwargs.get('subset_size', 16)     # blob analysis window size in pixels
    dx2=subset_size//2
    pixel_size = kwargs.get('pixel_size', 4.0)
    bounds = kwargs.get('bounds', [0.37, 0.63])
    bands = kwargs.get('bands', [5, 5, 5])        # bands for finding left minimum, mid (peak), and right minimum
    min_thr = kwargs.get('min_thr', 0.4)        #threshold for identifying 'good' transition (bottom < min_thr* top)
    transition_low_limit = kwargs.get('transition_low_limit', 0.0)
    transition_high_limit = kwargs.get('transition_high_limit', 10.0)
    save_data_xlsx = kwargs.get('save_data_xlsx', True)

    default_results_file_xlsx = os.path.join(data_dir, 'Dataset_2D_blob_analysis_results.xlsx')
    results_file_xlsx = kwargs.get('results_file_xlsx', default_results_file_xlsx)

    if use_DASK:
        verbose = False
        disp_res = False
    else:
        verbose = kwargs.get('verbose', False)
        disp_res = kwargs.get('disp_res', False)

    title = kwargs.get('title', '')
    nbins = kwargs.get('nbins', 64)
    if verbose:
        if sliding_evaluation_box:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will use sliding (linearly) evaluation box')
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Starting with box:  ', start_evaluation_box)
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Finishing with box: ', stop_evaluation_box)
        else:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will use fixed evaluation box: ', evaluation_box)

    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    mrc_mode = header.mode
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    header_dict = {}
    for record in header.dtype.names: # create dictionary from the header data
        if ('extra' not in record) and ('label' not in record):
            header_dict[record] = header[record]
    '''
    mode 0 -> uint8
    mode 1 -> int16
    mode 2 -> float32
    mode 4 -> complex64
    mode 6 -> uint16
    '''
    if mrc_mode==0:
        dt_mrc=np.uint8
    if mrc_mode==1:
        dt_mrc=np.int16
    if mrc_mode==2:
        dt_mrc=np.float32
    if mrc_mode==4:
        dt_mrc=np.complex64
    if mrc_mode==6:
        dt_mrc=np.uint16
    #print('mrc_mode={:d} '.format(mrc_mode), ', dt_mrc=', dt_mrc)

    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny
    evals = [xi_eval, xa_eval, yi_eval, ya_eval]
    
    frame_inds_default = np.arange(nz-1)+1
    frame_inds = np.array(kwargs.get("frame_inds", frame_inds_default))
    nf = frame_inds[-1]-frame_inds[0]+1
    if frame_inds[0]==0:
        frame_inds = frame_inds+1
    sample_frame_inds = [frame_inds[nf//10], frame_inds[nf//2], frame_inds[nf//10*9]]
    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will analyze 2D Blobs in {:d} frames'.format(len(frame_inds)))
        print('Will save the data into ' + results_file_xlsx)
    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    shape = np.shape(mrc_obj.data[frame_inds[nf//2], :, :])

    vmin, vmax = get_min_max_thresholds((mrc_obj.data[frame_inds[nf//2], yi_eval:ya_eval, xi_eval:xa_eval].astype(dt_mrc)).astype(float), thr_min=0.2, disp_res=False, save_res=False)
    threshold = kwargs.get('threshold', vmin/10.0)
    if verbose:
        print('Will use threshold : {:.4f}'.format(threshold))

    eval_bounds = set_eval_bounds(shape, evaluation_box,
        start_evaluation_box = start_evaluation_box,
        stop_evaluation_box = stop_evaluation_box,
        sliding_evaluation_box = sliding_evaluation_box,
        pad_edges = False,
        perform_transformation = False,
        tr_matr =  np.eye(3,3),
        frame_inds = frame_inds)
    
    padx = 0
    pady = 0
    xi = 0
    yi = 0
    shift_matrix = np.eye(3,3)
    inv_shift_matrix = np.eye(3,3)
    offsets = [xi, yi, padx, pady]
    kwargs['offsets'] = offsets

    mrc_papams_blob_analysis = []
    results_2D = []
    eval_bounds_df = pd.DataFrame(np.array(eval_bounds), columns = ['xi_eval', 'xa_eval', 'yi_eval', 'ya_eval'], index = None)
    eval_bounds_df.insert(0, 'Frame', frame_inds)

    for j, frame_ind in enumerate(tqdm(frame_inds, desc='Building the Parameter Sets Analyzing Resolution using Blobs ', display=verbose)):
        mrc_params_single = [mrc_filename, frame_ind, eval_bounds[j], offsets, invert_data, flipY, zbin_factor, min_sigma, max_sigma, threshold,  overlap, pixel_size, subset_size, bounds, bands, min_thr, transition_low_limit, transition_high_limit, nbins, verbose, disp_res, False]
        mrc_papams_blob_analysis.append(mrc_params_single)

    if use_DASK:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Blob Analysis, Using DASK distributed')
        futures = DASK_client.map(select_blobs_LoG_analyze_transitions_2D_mrc_stack, mrc_papams_blob_analysis, retries = DASK_client_retries)
        results_2D = DASK_client.gather(futures)

    else:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Blob Analysis, Using Local Computation')
        for j, mrc_params_single in enumerate(tqdm(mrc_papams_blob_analysis, desc='Analyzing Resolution using Blobs', display = True)):
            results_2D.append(select_blobs_LoG_analyze_transitions_2D_mrc_stack(mrc_params_single))

    frame_inds = np.concatenate(np.array(results_2D, dtype=object)[:, 0], axis=0)
    error_flags = np.concatenate(np.array(results_2D, dtype=object)[:, 1], axis=0)
    blobs_LoG = np.concatenate(np.array(results_2D, dtype=object)[:, 2], axis=0)
    tr_results = np.concatenate(np.array(results_2D, dtype=object)[:, 3], axis=0)

    if save_data_xlsx:
        xlsx_writer = pd.ExcelWriter(results_file_xlsx, engine='xlsxwriter')
        trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
        columns=['Frame', 'Y', 'X', 'R', 'Amp',
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

        if verbose:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the results into file:  ' + results_file_xlsx)

        transition_results = pd.DataFrame(np.column_stack((frame_inds, blobs_LoG, tr_results, error_flags)), columns = columns, index = None)
        transition_results.to_excel(xlsx_writer, index=None, sheet_name='Transition analysis results')
        kwargs_info = pd.DataFrame([kwargs]).T # prepare to be save in transposed format
        kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')
        eval_bounds_df.to_excel(xlsx_writer, index=None, sheet_name='Eval Bounds Info')
        fexts =['_{:.0f}{:.0f}pts'.format(bounds[0]*100, bounds[1]*100), '_{:.0f}{:.0f}slp'.format(bounds[0]*100, bounds[1]*100)]
        sheet_names = ['{:.0f}%-{:.0f}% summary (pts)'.format(bounds[0]*100, bounds[1]*100),
            '{:.0f}%-{:.0f}% summary (slopes)'.format(bounds[0]*100, bounds[1]*100)]
        #xlsx_writer.save()
        xlsx_writer.close()

    # return results_2D
    return results_file_xlsx, frame_inds, error_flags, blobs_LoG, tr_results


def select_blobs_LoG_analyze_transitions_2D_mrc_stack(params):
    '''
    DASK wrapper for select_blobs_LoG_analyze_transitions
    Finds blobs in the given grayscale image using Laplasian of Gaussians (LoG). gleb.shtengel@gmail.com 02/2024
    
    Parameters:
    params = list of: [mrc_filename, frame_ind, eval_bounds_single_frame, offsets, invert_data, flipY, zbin_factor, min_sigma, max_sigma, threshold,  overlap, pixel_size, subset_size, bounds, bands,
        min_thr, transition_low_limit, transition_high_limit, nbins, verbose, disp_res, save_data]
    mrc_filename
    frame_ind : index of frame
    eval_bounds_single_frame
    offsets = [xi, yi, padx, pady]
    invert_data : boolean
        If True - the data is inverted
    flipY
    zbin_factor
    min_sigma : float
        min sigma (in pixel units) for Gaussian kernel in LoG search.
    max_sigma : float
        min sigma (in pixel units) for Gaussian kernel in LoG search.
    threshold : float
        threshold for LoG search. The absolute lower bound for scale space maxima. Local maxima smaller
        than threshold are ignored. Reduce this to detect blobs with less intensities. 
    overlap : float
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than 'overlap', the smaller blob is eliminated.    
    pixel_size : float
        pixel size in nm.
    subset_size : int
        subset size (pixels) for blob / transition analysis
    bounds : lists
        List of transition limits.
    bands : list of 3 ints
        list of three ints for the averaging bands for determining the left min, peak, and right min of the cross-section profile.
    min_thr : float
        threshold for identifying a 'good' transition (bottom < min_thr* top)
    transition_low_limit : float
        error flag is incremented by 4 if the determined transition distance is below this value.
    transition_high_limit : float
        error flag is incremented by 8 if the determined transition distance is above this value.
    title : str
        title.
    nbins : int
        bins for histogram
    verbose
    disp_res
    save_data
    
    Returns: XY_transitions
        XY_transitions with error_flag=0
    '''
    [mrc_filename, frame_ind, eval_bounds_single_frame, offsets, invert_data, flipY, zbin_factor, min_sigma, max_sigma, threshold,  overlap, pixel_size, subset_size, bounds, bands, min_thr, transition_low_limit, transition_high_limit, nbins, verbose, disp_res, save_data] = params
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    mrc_mode = header.mode
    if mrc_mode==0:
        dt_mrc=np.uint8
    if mrc_mode==1:
        dt_mrc=np.int16
    if mrc_mode==2:
        dt_mrc=np.float32
    if mrc_mode==4:
        dt_mrc=np.complex64
    if mrc_mode==6:
        dt_mrc=np.uint16
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    
    frame = mrc_obj.data[frame_ind, :, :].astype(dt_mrc).astype(float)            
    shape = np.shape(frame)

    padx = 0
    pady = 0
    xi = 0
    yi = 0
    shift_matrix = np.eye(3,3)
    inv_shift_matrix = np.eye(3,3)
    xsz = shape[1] + padx
    ysz = shape[0] + pady

    xa = xi+shape[1]
    ya = yi+shape[0]
    xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds_single_frame
    
    frame_img = np.zeros((ysz, xsz), dtype=float)
    frame_eval = np.zeros(((ya_eval-yi_eval), (xa_eval-xi_eval)), dtype=float)

    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will analyze a subset of ', mrc_filename)

    for j in np.arange(zbin_factor):
        if j>0:
            frame = mrc_obj.data[frame_inds[frame_ind+j], :, :].astype(dt_mrc).astype(float)
        if invert_data:
            if dt_mrc==np.uint8:
                frame_img[yi:ya, xi:xa] = 255.0 - frame
            else:
                frame_img[yi:ya, xi:xa] = np.negative(frame)
        else:
            frame_img[yi:ya, xi:xa]  = frame

        if flipY:
            frame_img = np.flip(frame_img, axis=0)

        frame_eval += frame_img[yi_eval:ya_eval, xi_eval:xa_eval]
    if zbin_factor > 1:
        frame_eval /= zbin_factor
    if verbose:
        print('Subset shape: ', np.shape(frame_eval))

    fname_root = os.path.splitext(os.path.split(mrc_filename)[1])[0]
    fname_base = os.path.split(mrc_filename)[0]
    res_png_fname = os.path.join(fname_base, fname_root + '_resolution_results.png')
    examples_png_fname = os.path.join(fname_base, fname_root + '_blob_examples.png')
    results_file_xlsx = os.path.join(fname_base, fname_root + '_resolution_results.xlsx')

    kwargs = {'min_sigma' : min_sigma,
    'max_sigma' : max_sigma,
    'threshold' : threshold,
    'overlap' : overlap,
    'subset_size' : subset_size,
    'pixel_size' : pixel_size,
    'bounds' : bounds,
    'bands' : bands,
    'min_thr' : min_thr,
    'transition_low_limit' : transition_low_limit,
    'transition_high_limit' : transition_high_limit,
    'verbose' : verbose,
    'disp_res' : disp_res,
    'title' : ' ',
    'nbins' : nbins,
    'save_data_xlsx' : save_data,
    'results_file_xlsx' : results_file_xlsx}

    results_file_xlsx, blobs_LoG, error_flags, tr_results, hst_datas =  select_blobs_LoG_analyze_transitions(frame_eval, **kwargs)
    if save_data and disp_res:
        res = plot_blob_map_and_results_single_image(frame_eval, results_file_xlsx, save_png=True)
        res = plot_blob_examples_single_image(frame_eval, results_file_xlsx, save_png=True)

    tr_results_arr = np.array(tr_results)
    frame_ind_arr = np.full(len(error_flags), frame_ind, dtype=int)
    Xpt1 = tr_results_arr[error_flags==0, 1]
    Xpt2 = tr_results_arr[error_flags==0, 2]
    Ypt1 = tr_results_arr[error_flags==0, 3]
    Ypt2 = tr_results_arr[error_flags==0, 4]
    XY_transitions = np.array([Xpt1, Xpt2, Ypt1, Ypt2]).T
    return  frame_ind_arr, error_flags, blobs_LoG, tr_results_arr


def mrc_stack_plot_2D_blob_examples(results_xlsx, **kwargs):
    '''
    Generates a figure with blob examples based on xlsx file created by mrc_stack_estimate_resolution_blobs_2D
    mrc_stack_estimate_resolution_blobs_2D analyzes the transitions in the image, uses select_blobs_LoG_analyze_transitions(frame_eval, **kwargs). gleb.shtengel@gmail.com  04/2025 

    Parameters:
    ---------
    results_file_xlsx : file name of Excel workbook of the results created by mrc_stack_estimate_resolution_blobs_2D
    
    kwargs:
    ---------
    save_png : boolean
        save the image into PNG file. Default is False.
    save_fname : string
        file name for to save the image
    nbins : int
        bins for histogram
    verbose : boolean
        print the outputs. Default is False
    '''
    save_png = kwargs.get('save_png', False)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_2D_blob_examples.png'))
    verbose = kwargs.get('verbose', False)

    eval_bounds_info = pd.read_excel(results_xlsx, sheet_name='Eval Bounds Info')
    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    pixel_size = saved_kwargs.get("pixel_size", 1.0)
    subset_size = saved_kwargs.get("subset_size", 2.0)
    dx2 = subset_size//2
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    bands = saved_kwargs.get("bands", [3, 2, 3])
    image_name = saved_kwargs.get("image_name", 'ImageA')
    perform_transformation =  saved_kwargs.get("perform_transformation", False)
    pad_edges =  saved_kwargs.get("pad_edges", True)
    ftype =  saved_kwargs.get("ftype", 0)
    zbin_factor =  saved_kwargs.get("zbin_factor", 1)
    flipY = saved_kwargs.get("flipY", False)
    invert_data = saved_kwargs.get("invert_data", False)
    int_order =  saved_kwargs.get("int_order", 1)
    offsets =  saved_kwargs.get("offsets", [0, 0, 0, 0])
    mrc_filename = saved_kwargs['mrc_filename']
    mrc_obj = mrcfile.mmap(mrc_filename, mode='r')
    header = mrc_obj.header
    mrc_mode = header.mode
    nx, ny, nz = np.int32(header['nx']), np.int32(header['ny']), np.int32(header['nz'])
    header_dict = {}
    for record in header.dtype.names: # create dictionary from the header data
        if ('extra' not in record) and ('label' not in record):
            header_dict[record] = header[record]
    '''
    mode 0 -> uint8
    mode 1 -> int16
    mode 2 -> float32
    mode 4 -> complex64
    mode 6 -> uint16
    '''
    if mrc_mode==0:
        dt_mrc=np.uint8
    if mrc_mode==1:
        dt_mrc=np.int16
    if mrc_mode==2:
        dt_mrc=np.float32
    if mrc_mode==4:
        dt_mrc=np.complex64
    if mrc_mode==6:
        dt_mrc=np.uint16
    #print('mrc_mode={:d} '.format(mrc_mode), ', dt_mrc=', dt_mrc)
    
    xs=16.0
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
    X_selected = np.array(X)[error_flags==0]
    Y_selected = np.array(Y)[error_flags==0]
    frames = int_results['Frame']
    frames_selected = np.array(frames)[error_flags==0]
    Xs = np.concatenate((X_selected[0:3], X_selected[-3:]))
    Ys = np.concatenate((Y_selected[0:3], Y_selected[-3:]))
    Fs = np.concatenate((frames_selected[0:3], frames_selected[-3:]))

    xt = 0.0
    yt = 1.5 
    clr_x = 'green'
    clr_y = 'blue'
    fig, axs = plt.subplots(4,3, figsize=(xs, ys))
    fig.subplots_adjust(left=0.02, bottom=0.04, right=0.99, top=0.99, wspace=0.15, hspace=0.12)

    ax_maps = [axs[0,0], axs[0,1], axs[0,2], axs[2,0], axs[2,1], axs[2,2]]
    ax_profiles = [axs[1,0], axs[1,1], axs[1,2], axs[3,0], axs[3,1], axs[3,2]]

    for j, x in enumerate(tqdm(Xs, desc='Generating images/plots for the sample 2D blobs')):
        local_ind = int(np.argwhere(np.array(eval_bounds_info['Frame']) == Fs[j]))
        local_eval_bounds_info = eval_bounds_info[eval_bounds_info['Frame'] == Fs[j]]
        #print(Fs[j], fl, local_ind)
        yi_eval = local_eval_bounds_info['yi_eval'].values[0]
        ya_eval = local_eval_bounds_info['ya_eval'].values[0]
        xi_eval = local_eval_bounds_info['xi_eval'].values[0]
        xa_eval = local_eval_bounds_info['xa_eval'].values[0]

        frame = mrc_obj.data[Fs[j], :, :].astype(dt_mrc).astype(float)
        shape = np.shape(frame)
        
        xi=0
        yi=0
        xsz = shape[1]
        ysz = shape[0]
        xa = xi+xsz
        ya = yi+ysz
  
        frame_img = np.zeros((ysz, xsz), dtype=float)
        frame_eval = np.zeros(((ya_eval-yi_eval), (xa_eval-xi_eval)), dtype=float)

        for jk in np.arange(zbin_factor):
            if jk>0:
                local_ind = int(np.squeeze(np.argwhere(np.array(fls_info['Frame']) == Fs[j+jk])))
                local_eval_bounds_info = eval_bounds_info[eval_bounds_info['Frame'] ==Fs[j+jk]]
                frame = mrc_obj.data[Fs[j+jk], :, :].astype(dt_mrc).astype(float)
                yi_eval = local_eval_bounds_info['yi_eval'].values[0]
                ya_eval = local_eval_bounds_info['ya_eval'].values[0]
                xi_eval = local_eval_bounds_info['xi_eval'].values[0]
                xa_eval = local_eval_bounds_info['xa_eval'].values[0]
            if invert_data:
                frame_img[yi:ya, xi:xa] = np.negative(frame)
            else:
                frame_img[yi:ya, xi:xa]  = frame
            if flipY:
                frame_img = np.flip(frame_img, axis=0)

            frame_eval += frame_img[yi_eval:ya_eval, xi_eval:xa_eval]
        if zbin_factor > 1:
            frame_eval /= zbin_factor
        
        y = Ys[j]
        xx = int(x)
        yy = int(y)
        subset = frame_eval[yy-dx2:yy+dx2, xx-dx2:xx+dx2]
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
        #print(np.shape(amp_x), np.shape(amp_y), np.shape(amp_z))
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


##########################################
#         TIF stack analysis functions
##########################################

def show_eval_box_tif_stack(tif_filename, **kwargs):
    '''
    Read tif stack and display the eval box for each frame from the list.
    ©G.Shtengel, 08/2022. gleb.shtengel@gmail.com

    Parameters
    ---------
    tif_filename : str
        File name (full path) of the tif stack to be analyzed
     
    kwargs:
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    ax : matplotlib ax artist
        if provided, the data is exported to external ax object.
    frame_inds : array
        List of frame indices to display the evaluation box. If not provided, three frames will be used:
        [nz//10,  nz//2, nz//10*9] where nz is number of frames in tif stack
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    box_linewidth : float
        linewidth for the box outline. Default is 1.0
    box_color : color
        color for the box outline. Default is yellow
    invert_data : Boolean
    '''
    tif_filename = os.path.normpath(tif_filename)

    Sample_ID = kwargs.get("Sample_ID", '')
    save_res_png  = kwargs.get("save_res_png", True )
    save_filename = kwargs.get("save_filename", tif_filename )
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    box_linewidth = kwargs.get("box_linewidth", 1.0)
    box_color = kwargs.get("box_color", 'yellow')
    invert_data =  kwargs.get("invert_data", False)
    ax = kwargs.get("ax", '')
    plot_internal = (ax == '')

    with tiff.TiffFile(tif_filename) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        #print(tif_tags)
    try:
        shape = eval(tif_tags['ImageDescription'])
        try:
            nz, ny, nx = shape['shape']
        except:
            nx = eval(tif_tags['ImageWidth'])
            ny = eval(tif_tags['ImageLength'])
            nz = np.int(tif_tags['ImageDescription'].split('images=')[1].split('\nslices')[0])
    except:
        try:
            shape = eval(tif_tags['image_description'])
            nz, ny, nx = shape['shape']
        except:
            fr0 = tiff.imread(tif_filename, key=0)
            ny, nx = np.shape(fr0)
            try:
                nz = eval(tif_tags['nimages'])
            except:
                nz = np.int(tif_tags['ImageDescription'].split('images=')[1].split('\nslices')[0])

    frame_inds = kwargs.get("frame_inds", [nz//10,  nz//2, nz//10*9] )
    
    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny

    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0

    for fr_ind in frame_inds: 
        #eval_frame = (tif.data[fr_ind, :, :].astype(dt)).astype(float)
        eval_frame = tiff.imread(tif_filename, key=fr_ind).astype(float)

        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*fr_ind//nz
            yi_eval = start_evaluation_box[0] + dy_eval*fr_ind//nz
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny

        if plot_internal:
            fig, ax = plt.subplots(1,1, figsize = (10.0, 11.0*ny/nx))
        dmin, dmax = get_min_max_thresholds(eval_frame[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
        if invert_data:
            ax.imshow(eval_frame, cmap='Greys_r', vmin=dmin, vmax=dmax)
        else:
            ax.imshow(eval_frame, cmap='Greys', vmin=dmin, vmax=dmax)
        ax.grid(True, color = "cyan")
        ax.set_title(Sample_ID + ' '+tif_filename +',  frame={:d}'.format(fr_ind))
        rect_patch = patches.Rectangle((xi_eval,yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2,
            linewidth=box_linewidth, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect_patch)
        if save_res_png  and plot_internal:
            fname = os.path.splitext(save_filename)[0] + '_frame_{:d}_evaluation_box.png'.format(fr_ind)
            fig.savefig(fname, dpi=300)


def evaluate_registration_two_frames_tif(params_tif):
    '''
    Helper function used by DASK routine. Analyzes registration between two frames.
    ©G.Shtengel, 08/2022. gleb.shtengel@gmail.com

    Parameters:
    params_tif : list of tif_filename, fr, evals
    tif_filename  : string
        full path to tif filename
    fr : int
        Index of the SECOND frame
    evals :  list of image bounds to be used for evaluation exi_eval, xa_eval, yi_eval, ya_eval, save_frame_png, filename_frame_png


    Returns:
    image_nsad, image_ncc, image_mi   : float, float, float

    '''
    tif_filename, fr, invert_data, evals, save_frame_png, filename_frame_png = params_tif
    tif_filename = os.path.normpath(tif_filename)
    filename_frame_png = os.path.normpath(filename_frame_png)
    xi_eval, xa_eval, yi_eval, ya_eval = evals
    
    frame0 = tiff.imread(tif_filename, key=int(fr-1)).astype(float)
    frame1 = tiff.imread(tif_filename, key=int(fr)).astype(float)
    
    if invert_data:
        prev_frame = -1.0 * frame0[yi_eval:ya_eval, xi_eval:xa_eval]
        curr_frame = -1.0 * frame1[yi_eval:ya_eval, xi_eval:xa_eval]
    else:
        prev_frame = frame0[yi_eval:ya_eval, xi_eval:xa_eval]
        curr_frame = frame1[yi_eval:ya_eval, xi_eval:xa_eval]
    fr_mean = np.abs(curr_frame/2.0 + prev_frame/2.0)
    #image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)-np.amin(fr_mean))
    image_nsad =  np.mean(np.abs(curr_frame-prev_frame))/(np.mean(fr_mean)-np.amin(fr_mean))
    image_ncc = Two_Image_NCC_SNR(curr_frame, prev_frame)[0]
    image_mi = mutual_information_2d(prev_frame.ravel(), curr_frame.ravel(), sigma=1.0, bin=2048, normalized=True)
    if save_frame_png:
        yshape, xshape = frame0.shape
        fig, ax = plt.subplots(1,1, figsize=(3.0*xshape/yshape, 3))
        fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
        dmin, dmax = get_min_max_thresholds(frame0[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
        if invert_data:
            ax.imshow(frame0, cmap='Greys_r', vmin=dmin, vmax=dmax)
        else:
            ax.imshow(frame0, cmap='Greys', vmin=dmin, vmax=dmax)
        #ax.text(0.06, 0.95, 'Frame={:d},  NSAD={:.3f},  NCC={:.3f},  NMI={:.3f}'.format(fr, image_nsad, image_ncc, image_mi), color='red', transform=ax.transAxes, fontsize=12)
        ax.text(0.02, 0.95, 'Frame={:d}'.format(fr), color='red', transform=ax.transAxes, fontsize=11)
        ax.text(0.50, 0.95, 'NSAD={:.3f}'.format(image_nsad), color='red', transform=ax.transAxes, fontsize=11)
        ax.text(0.50, 0.85, 'NCC={:.3f}'.format(image_ncc), color='red', transform=ax.transAxes, fontsize=11)
        ax.text(0.50, 0.75, 'NMI={:.3f}'.format(image_mi), color='red', transform=ax.transAxes, fontsize=11)
        rect_patch = patches.Rectangle((xi_eval, yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2, linewidth=1.0, edgecolor='yellow',facecolor='none')
        ax.add_patch(rect_patch)
        ax.axis('off')
        fig.savefig(filename_frame_png, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)   # save the figure to file
        plt.close(fig)

    return image_nsad, image_ncc, image_mi


def analyze_tif_stack_registration(tif_filename, **kwargs):
    '''
    Read MRC stack and analyze registration - calculate NSAD, NCC, and MI.
    ©G.Shtengel, 08/2022. gleb.shtengel@gmail.com

    Parameters
    ---------
    tif_filename : str
        File name (full path) of the mrc stack to be analyzed

    kwargs:
    DASK_client : DASK client. If set to empty string '' (default), local computations are performed
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames
    invert_data : boolean
        If True, the data will be inverted
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    save_sample_frames_png : boolean
        If True, sample frames with superimposed eval box and registration analysis data will be saved into png files
    sliding_evaluation_box : boolean
        if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box
    start_evaluation_box : list of 4 int
        see above
    stop_evaluation_box : list of 4 int
        see above
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    save_filename : str
        Path to the filename to save the results. If empty, tif_filename+'_RegistrationQuality.xlsx' will be used

    Returns reg_summary : PD data frame, registration_summary_xlsx : path to summary XLSX spreadsheet file
    '''
    tif_filename = os.path.normpath(tif_filename)
    
    Sample_ID = kwargs.get("Sample_ID", '')
    DASK_client = kwargs.get('DASK_client', '')
    DASK_client_retries = kwargs.get('DASK_client_retries', 3)
    use_DASK, status_update_address = check_DASK(DASK_client)
 
    invert_data =  kwargs.get("invert_data", False)
    save_res_png  = kwargs.get("save_res_png", True )
    save_filename = kwargs.get("save_filename", tif_filename )
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
    start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
    stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
    save_sample_frames_png = kwargs.get("save_sample_frames_png", True)
    registration_summary_xlsx = save_filename.replace('.mrc', '_RegistrationQuality.xlsx')

    if sliding_evaluation_box:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will use sliding (linearly) evaluation box')
        print('   Starting with box:  ', start_evaluation_box)
        print('   Finishing with box: ', stop_evaluation_box)
    else:
        print('Will use fixed evaluation box: ', evaluation_box)

    with tiff.TiffFile(tif_filename) as tif:
        tif_tags = {}
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            tif_tags[name] = value
        #print(tif_tags)
    try:
        shape = eval(tif_tags['ImageDescription'])
        nz, ny, nx = shape['shape']
    except:
        try:
            shape = eval(tif_tags['image_description'])
            try:
                nz, ny, nx = shape['shape']
            except:
                nx = eval(tif_tags['ImageWidth'])
                ny = eval(tif_tags['ImageLength'])
                nz = np.int(tif_tags['ImageDescription'].split('images=')[1].split('\nslices')[0])
        except:
            fr0 = tiff.imread(tif_filename, key=0)
            try:
                ny, nx = np.shape(fr0)
                nz = eval(tif_tags['nimages'])
            except:
                nz, ny, nx = np.shape(fr0)

    header_dict = {'nx' : nx, 'ny' : ny, 'nz' : nz }

    xi_eval = evaluation_box[2]
    if evaluation_box[3] > 0:
        xa_eval = xi_eval + evaluation_box[3]
    else:
        xa_eval = nx
    yi_eval = evaluation_box[0]
    if evaluation_box[1] > 0:
        ya_eval = yi_eval + evaluation_box[1]
    else:
        ya_eval = ny
    evals = [xi_eval, xa_eval, yi_eval, ya_eval]
    
    frame_inds_default = np.arange(nz-1)+1
    frame_inds = np.array(kwargs.get("frame_inds", frame_inds_default))
    nf = frame_inds[-1]-frame_inds[0]+1
    if frame_inds[0]==0:
        frame_inds = frame_inds+1
    sample_frame_inds = [frame_inds[nf//10], frame_inds[nf//2], frame_inds[nf//10*9]]
    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will analyze regstrations in {:d} frames'.format(len(frame_inds)))
    print('Will save the data into ' + registration_summary_xlsx)
    if sliding_evaluation_box:
        dx_eval = stop_evaluation_box[2]-start_evaluation_box[2]
        dy_eval = stop_evaluation_box[0]-start_evaluation_box[0]
    else:
        dx_eval = 0
        dy_eval = 0
    
    params_tif_mult = []
    xi_evals = np.zeros(nf, dtype=np.int16)
    xa_evals = np.zeros(nf, dtype=np.int16)
    yi_evals = np.zeros(nf, dtype=np.int16)
    ya_evals = np.zeros(nf, dtype=np.int16)
    for j, fr in enumerate(frame_inds):
        if sliding_evaluation_box:
            xi_eval = start_evaluation_box[2] + dx_eval*(fr-frame_inds[0])//nf
            yi_eval = start_evaluation_box[0] + dy_eval*(fr-frame_inds[0])//nf
            if start_evaluation_box[3] > 0:
                xa_eval = xi_eval + start_evaluation_box[3]
            else:
                xa_eval = nx
            if start_evaluation_box[1] > 0:
                ya_eval = yi_eval + start_evaluation_box[1]
            else:
                ya_eval = ny
            evals = [xi_eval, xa_eval, yi_eval, ya_eval]
        xi_evals[j] = xi_eval
        xa_evals[j] = xa_eval
        yi_evals[j] = yi_eval
        ya_evals[j] = ya_eval
        if fr in sample_frame_inds:
            save_frame_png = save_sample_frames_png
            filename_frame_png = os.path.splitext(save_filename)[0]+'_sample_image_frame{:d}.png'.format(fr)
        else:
            save_frame_png = False
            filename_frame_png = os.path.splitext(save_filename)[0]+'_sample_image_frame.png'
        params_tif_mult.append([tif_filename, fr, invert_data, evals, save_frame_png, filename_frame_png])
        
    if use_DASK:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
        futures = DASK_client.map(evaluate_registration_two_frames_tif, params_tif_mult, retries = DASK_client_retries)
        dask_results = DASK_client.gather(futures)
        image_nsad = np.array([res[0] for res in dask_results])
        image_ncc = np.array([res[1] for res in dask_results])
        image_mi = np.array([res[2] for res in dask_results])
    else:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using Local Computation')
        image_nsad = np.zeros((nf), dtype=float)
        image_ncc = np.zeros((nf), dtype=float)
        image_mi = np.zeros((nf), dtype=float)
        results = []
        for params_tif_mult_pair in tqdm(params_tif_mult, desc='Evaluating frame registration: '):
            results.append(evaluate_registration_two_frames_tif(params_tif_mult_pair))
        image_nsad = np.array([res[0] for res in results])
        image_ncc = np.array([res[1] for res in results])
        image_mi = np.array([res[2] for res in results])
    
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)] 
    #image_ncc = image_ncc[1:-1]
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_mi), np.median(image_mi), np.std(image_mi)]

    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the Registration Quality Statistics into the file: ', registration_summary_xlsx)
    xlsx_writer = pd.ExcelWriter(registration_summary_xlsx, engine='xlsxwriter')
    columns=['Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval', 'NSAD', 'NCC', 'NMI']
    reg_summary = pd.DataFrame(np.vstack((frame_inds, xi_evals, xa_evals, yi_evals, ya_evals, image_nsad, image_ncc, image_mi)).T, columns = columns, index = None)
    reg_summary.to_excel(xlsx_writer, index=None, sheet_name='Registration Quality Statistics')
    Stack_info = pd.DataFrame([{'Stack Filename' : tif_filename, 'Sample_ID' : Sample_ID, 'invert_data' : invert_data}]).T # prepare to be save in transposed format
    header_info = pd.DataFrame([header_dict]).T
    #Stack_info = Stack_info.append(header_info)  append has been removed from pandas as of 2.0.0, use concat instead
    Stack_info = pd.concat([Stack_info, header_info], axis=1)
    Stack_info.to_excel(xlsx_writer, header=False, sheet_name='Stack Info')
    #xlsx_writer.save()
    xlsx_writer.close()

    return reg_summary, registration_summary_xlsx


##########################################
#         helper functions for analysis of FiJi registration
##########################################

def read_transformation_matrix_from_xf_file(xf_filename):
    '''
    Reads transformation matrix created by FiJi-based workflow from *.xf file. ©G.Shtengel 10/2022 gleb.shtengel@gmail.com
    
    Parameters:
    xf_filename : str
        Full path to *.xf file containing the transformation matrix data

    Returns:
    transformation_matrix : array
    '''
    npdt_recalled = pd.read_csv(xf_filename, sep = '  ', header = None)
    tr = npdt_recalled.to_numpy()
    transformation_matrix = np.zeros((len(tr), 3, 3))
    transformation_matrix[:, 0, 0:2] = tr[:,0:2]
    transformation_matrix[:, 0, 2] = tr[:,6]
    transformation_matrix[:, 1, 0:2] = tr[:,2:4]
    transformation_matrix[:, 1, 2] = tr[:,9]
    transformation_matrix[:, 2, 2] = np.ones((len(tr)))   
    return transformation_matrix

def analyze_transformation_matrix(transformation_matrix, xf_filename):
    '''
    Analyzes the transformation matrix created by FiJi-based workflow. ©G.Shtengel 10/2022 gleb.shtengel@gmail.com
    
    Parameters:
    transformation_matrix : array
        Transformation matrix (read by read_transformation_matrix_from_xf_file above).
    xf_filename : str
        Full path to *.xf file containing the transformation matrix data

    Returns:
    tr_matr_cum : array
    Cumulative transformation matrix
    '''
    Xshift_orig = transformation_matrix[:, 0, 2]
    Yshift_orig = transformation_matrix[:, 1, 2]
    Xscale_orig = transformation_matrix[:, 0, 0]
    Yscale_orig = transformation_matrix[:, 1, 1]
    tr_matr_cum = transformation_matrix.copy()

    prev_mt = np.eye(3,3)
    for j, cur_mt in enumerate(tqdm(transformation_matrix, desc='Calculating Cummilative Transformation Matrix')):
        if np.any(np.isnan(cur_mt)):
            print('Frame: {:d} has ill-defined transformation matrix, will use identity transformation instead:'.format(j))
            print(cur_mt)
        else:
            prev_mt = np.matmul(cur_mt, prev_mt)
        tr_matr_cum[j] = prev_mt
    # Now insert identity matrix for the zero frame which does not need to be trasformed
    tr_matr_cum_orig = tr_matr_cum.copy()

    s00_cum_orig = tr_matr_cum[:, 0, 0].copy()
    s11_cum_orig = tr_matr_cum[:, 1, 1].copy()
    s01_cum_orig = tr_matr_cum[:, 0, 1].copy()
    s10_cum_orig = tr_matr_cum[:, 1, 0].copy()
    
    Xshift_cum_orig = tr_matr_cum_orig[:, 0, 2]
    Yshift_cum_orig = tr_matr_cum_orig[:, 1, 2]


    #print('Recalculating Shifts')
    s00_cum_orig = tr_matr_cum[:, 0, 0]
    s11_cum_orig = tr_matr_cum[:, 1, 1]
    fr = np.arange(0, len(s00_cum_orig), dtype=float)
    s00_slp = -1.0 * (np.sum(fr)-np.dot(s00_cum_orig,fr))/np.dot(fr,fr) # find the slope of a linear fit with fiorced first scale=1
    s00_fit = 1.0 + s00_slp * fr
    s00_cum_new = s00_cum_orig + 1.0 - s00_fit
    s11_slp = -1.0 * (np.sum(fr)-np.dot(s11_cum_orig,fr))/np.dot(fr,fr) # find the slope of a linear fit with fiorced first scale=1
    s11_fit = 1.0 + s11_slp * fr
    s11_cum_new = s11_cum_orig + 1.0 - s11_fit
    
    s01_slp = np.dot(s01_cum_orig,fr)/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
    s01_fit = s01_slp * fr
    s01_cum_new = s01_cum_orig - s01_fit
    s10_slp = np.dot(s10_cum_orig,fr)/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
    s10_fit = s10_slp * fr
    s10_cum_new = s10_cum_orig - s10_fit

    Xshift_cum = tr_matr_cum[:, 0, 2]
    Yshift_cum = tr_matr_cum[:, 1, 2]

    subtract_linear_fit=True

    # Subtract linear trend from offsets
    if subtract_linear_fit:
        fr = np.arange(0, len(Xshift_cum) )
        pX = np.polyfit(fr, Xshift_cum, 1)
        Xfit = np.polyval(pX, fr)
        pY = np.polyfit(fr, Yshift_cum, 1)
        Yfit = np.polyval(pY, fr)
        Xshift_residual = Xshift_cum - Xfit
        Yshift_residual = Yshift_cum - Yfit
    else:
        Xshift_residual = Xshift_cum.copy()
        Yshift_residual = Yshift_cum.copy()

    # define new cum. transformation matrix where the offests may have linear slopes subtracted
    tr_matr_cum_residual = tr_matr_cum.copy()
    tr_matr_cum_residual[:, 0, 2] = Xshift_residual
    tr_matr_cum_residual[:, 1, 2] = Yshift_residual
    tr_matr_cum_residual[:, 0, 0] = s00_cum_new
    tr_matr_cum_residual[:, 1, 1] = s11_cum_new
    tr_matr_cum_residual[:, 0, 1] = s01_cum_new
    tr_matr_cum_residual[:, 1, 0] = s10_cum_new

    fs = 12
    fig5, axs5 = plt.subplots(3,3, figsize=(18, 12), sharex=True)
    fig5.subplots_adjust(left=0.15, bottom=0.08, right=0.99, top=0.94)

    # plot scales
    axs5[0, 0].plot(Xscale_orig, 'r', label = 'Sxx fr.-to-fr.')
    axs5[0, 0].plot(Yscale_orig, 'b', label = 'Syy fr.-to-fr.')
    axs5[0, 0].set_title('Frame-to-Frame Scale Change', fontsize = fs + 1)
    axs5[1, 0].plot(tr_matr_cum_orig[:, 0, 0], 'r', linestyle='dotted', label = 'Sxx cum.')
    axs5[1, 0].plot(tr_matr_cum_orig[:, 1, 1], 'b', linestyle='dotted', label = 'Syy cum.')
    axs5[1, 0].plot(s00_fit, 'r', label = 'Sxx cum. - lin. fit')
    axs5[1, 0].plot(s11_fit, 'b', label = 'Syy cum. - lin. fit')
    axs5[1, 0].set_title('Cumulative Scale', fontsize = fs + 1)
    axs5[2, 0].plot(tr_matr_cum_residual[:, 0, 0], 'r', label = 'Sxx cum. - residual')
    axs5[2, 0].plot(tr_matr_cum_residual[:, 1, 1], 'b', label = 'Syy cum. - residual')
    axs5[2, 0].set_title('Residual Cumulative Scale', fontsize = fs + 1)
    axs5[2, 0].set_xlabel('Frame', fontsize = fs + 1)
    
    # plot shears
    axs5[0, 1].plot(transformation_matrix[:, 0, 1], 'r', label = 'Sxy fr.-to-fr.')
    axs5[0, 1].plot(transformation_matrix[:, 1, 0], 'b', label = 'Syx fr.-to-fr.')
    axs5[0, 1].set_title('Frame-to-Frame Shear', fontsize = fs + 1)
    axs5[1, 1].plot(tr_matr_cum_orig[:, 0, 1], 'r', linestyle='dotted', label = 'Sxy cum.')
    axs5[1, 1].plot(tr_matr_cum_orig[:, 1, 0], 'b', linestyle='dotted', label = 'Syx cum.')
    axs5[1, 1].plot(s01_fit, 'r', label = 'Sxy cum. - lin. fit')
    axs5[1, 1].plot(s10_fit, 'b', label = 'Syx cum. - lin. fit')
    axs5[1, 1].set_title('Cumulative Shear', fontsize = fs + 1)
    axs5[2, 1].plot(tr_matr_cum_residual[:, 0, 1], 'r', label = 'Sxy cum. - residual')
    axs5[2, 1].plot(tr_matr_cum_residual[:, 1, 0], 'b', label = 'Syx cum. - residual')
    axs5[2, 1].set_title('Residual Cumulative Shear', fontsize = fs + 1)
    axs5[2, 1].set_xlabel('Frame', fontsize = fs + 1)

    # plot shifts
    axs5[0, 2].plot(Xshift_orig, 'r', label = 'Tx fr.-to-fr.')
    axs5[0, 2].plot(Yshift_orig, 'b', label = 'Ty fr.-to-fr.')
    axs5[0, 2].set_title('Frame-to-Frame Shift', fontsize = fs + 1)
    axs5[1, 2].plot(Xshift_cum, 'r', linestyle='dotted', label = 'Tx cum.')
    axs5[1, 2].plot(Xfit, 'r', label = 'Tx cum. - lin. fit')
    axs5[1, 2].plot(Yshift_cum, 'b', linestyle='dotted', label = 'Ty cum.')
    axs5[1, 2].plot(Yfit, 'b', label = 'Ty cum. - lin. fit')
    axs5[1, 2].set_title('Cumulative Shift', fontsize = fs + 1)
    axs5[2, 2].plot(tr_matr_cum_residual[:, 0, 2], 'r', label = 'Tx cum. - residual')
    axs5[2, 2].plot(tr_matr_cum_residual[:, 1, 2], 'b', label = 'Ty cum. - residual')
    axs5[2, 2].set_title('Residual Cumulative Shift', fontsize = fs + 1)
    axs5[2, 2].set_xlabel('Frame', fontsize = fs + 1)

    for ax in axs5.ravel():
        ax.grid(True)
        ax.legend()
    fig5.suptitle(xf_filename, fontsize = fs + 2)
    fig5.savefig(xf_filename +'_Transform_Summary.png', dpi=300)
    return tr_matr_cum


##########################################
#         helper functions for results presentation
##########################################


def generate_report_mill_rate_xlsx(Mill_Rate_Data_xlsx, **kwargs):
    '''
    Generate Report Plot for mill rate evaluation from XLSX spreadsheet file. ©G.Shtengel 12/2022 gleb.shtengel@gmail.com
    
    Parameters:
    Mill_Rate_Data_xlsx : str
        Path to the XLSX spreadsheet file containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.
    
    kwargs:
    Mill_Volt_Rate_um_per_V : float
        Milling Voltage to Z conversion (µm/V). Default is 31.235258870176065.

    '''
    disp_res = kwargs.get('disp_res', False)
    if disp_res:
        print('Loading kwarg Data')
    saved_kwargs = read_kwargs_xlsx(Mill_Rate_Data_xlsx, 'kwargs Info', **kwargs)
    data_dir = saved_kwargs.get("data_dir", '')
    Sample_ID = saved_kwargs.get("Sample_ID", '')
    Saved_Mill_Volt_Rate_um_per_V = saved_kwargs.get("Mill_Volt_Rate_um_per_V", 31.235258870176065)
    Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", Saved_Mill_Volt_Rate_um_per_V)
    
    if disp_res:
        print('Loading Working Distance and Milling Y Voltage Data')
    try:
        int_results = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name='FIBSEM Data')
    except:
        int_results = pd.read_excel(Mill_Rate_Data_xlsx, sheet_name='Milling Rate Data')
    fr = int_results['Frame']
    WD = int_results['Working Distance (mm)']
    MillingYVoltage = int_results['Milling Y Voltage (V)']

    if disp_res:
        print('Generating Plot')
    fs = 12
    Mill_Volt_Rate_um_per_V = 31.235258870176065

    fig, axs = plt.subplots(2,1, figsize = (6,7), sharex=True)
    fig.subplots_adjust(left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.05, hspace=0.05)
    axs[0].plot(fr, WD, label='WD, Exp. Data', color='blue')
    axs[0].grid(True)
    axs[0].set_ylabel('Working Distance (mm)')
    #axs[0].set_xlim(xi, xa)
    WD_fit_coef = np.polyfit(fr, WD, 1)
    WD_fit=np.polyval(WD_fit_coef, fr)
    axs[0].plot(fr, WD_fit, label='Fit, slope = {:.2f} nm/line'.format(WD_fit_coef[0]*1.0e6), color='red')
    axs[0].legend(fontsize=12)

    axs[1].plot(fr, MillingYVoltage, label='Mill. Y Volt. Exp. Data', color='green')
    axs[1].grid(True)
    axs[1].set_ylabel('Milling Y Voltage (V)')
    MV_fit_coef = np.polyfit(fr, MillingYVoltage, 1)
    MV_fit=np.polyval(MV_fit_coef, fr)
    axs[1].plot(fr, MV_fit, label='Fit, slope = {:.3f} nm/line'.format(MV_fit_coef[0]*Mill_Volt_Rate_um_per_V*-1.0e3), color='orange')
    axs[1].legend(fontsize=12)
    axs[1].text(0.02, 0.05, 'Milling Voltage to Z conversion: {:.4f} µm/V'.format(Mill_Volt_Rate_um_per_V), transform=axs[1].transAxes, fontsize=12)
    axs[1].set_xlabel('Frame')
    ldm = 70
    data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]
    try:
        axs[0].text(-0.15, 1.05, Sample_ID + '    ' +  data_dir_short, fontsize = fs-2, transform=axs[0].transAxes)
    except:
        axs[0].text(-0.15, 1.05, data_dir_short, fontsize = fs-2, transform=axs[0].transAxes)
    fig.savefig(os.path.join(data_dir, Mill_Rate_Data_xlsx.replace('.xlsx','_Mill_Rate.png')), dpi=300)


def generate_report_ScanRate_EHT_xlsx(ScanRate_EHT_Data_xlsx, **kwargs):
    '''
    Generate Report Plot for Scan Rate and EHT from XLSX spreadsheet file. ©G.Shtengel 09/2023 gleb.shtengel@gmail.com
    
    Parameters:
    ScanRate_EHT_Data_xlsx : str
        Path to the XLSX spreadsheet file containing the Scan Rate and EHT data.
    
    kwargs:
    '''
    disp_res = kwargs.get('disp_res', False)
    if disp_res:
        print('Loading kwarg Data')
    saved_kwargs = read_kwargs_xlsx(ScanRate_EHT_Data_xlsx, 'kwargs Info', **kwargs)
    data_dir = saved_kwargs.get("data_dir", '')
    Sample_ID = saved_kwargs.get("Sample_ID", '')
    
    if disp_res:
        print('Loading Scan Rate and EHT Data')
    try:
        int_results = pd.read_excel(ScanRate_EHT_Data_xlsx, sheet_name='FIBSEM Data')
    except:
        int_results =[]
    fr = int_results['Frame']
    ScanRate = int_results['Scan Rate (Hz)']
    EHT = int_results['EHT (kV)']
    try:
        SEMSpecimenI = int_results['SEMSpecimenI (nA)']
    except:
        SEMSpecimenI = EHT*0.0

    if disp_res:
        print('Generating Plot')
    fs = 12

    fig, axs = plt.subplots(2,1, figsize = (6,7), sharex=True)
    fig.subplots_adjust(left=0.12, bottom=0.06, right=0.88, top=0.96, wspace=0.05, hspace=0.05)
    axs[0].plot(ScanRate/1000.0, 'b', label='Scan Rate')
    ax2 = axs[0].twinx()
    ax2.plot(EHT, 'r', linestyle='dashed', label='EHT')
    ax2.set_ylabel('EHT (kV)', color = 'red')
    ax2.legend(loc='upper right')
    
    axs[1].plot(SEMSpecimenI, 'g', label='SEM Specimen I')
    axs[1].set_xlabel('Frame')
    axs[0].set_ylabel('Scan Rate (kHz)', color = 'blue')
    axs[1].set_ylabel('SEM Specimen I (nA)', color='g')
    for ax in axs:
        ax.grid(True)
        ax.legend(loc='upper left')
    axs[0].set_title(data_dir)
    fig.savefig(os.path.join(data_dir, ScanRate_EHT_Data_xlsx.replace('.xlsx','_FIBSEM_Data_ScanRate_EHT.png')), dpi=300)


def generate_report_FOV_center_shift_xlsx(FOV_center_shift_xlsx, **kwargs):
    '''
    Generate Report Plot for FOV center shift from XLSX spreadsheet file. ©G.Shtengel 12/2022 gleb.shtengel@gmail.com
    
    Parameters:
    FOV_center_shift_xlsx : str
        Path to the XLSX spreadsheet file containing the Working Distance (WD), Milling Y Voltage (MV), and FOV center shifts data.
    
    kwargs:

    '''
    disp_res = kwargs.get('disp_res', False)
    if disp_res:
        print('Loading kwarg Data')
    saved_kwargs = read_kwargs_xlsx(FOV_center_shift_xlsx, 'kwargs Info', **kwargs)
    data_dir = saved_kwargs.get("data_dir", '')
    Sample_ID = saved_kwargs.get("Sample_ID", '')
    
    if disp_res:
        print('Loading FOV Center Location Data')
    try:
        int_results = pd.read_excel(FOV_center_shift_xlsx, sheet_name='FIBSEM Data')
    except:
        int_results = pd.read_excel(FOV_center_shift_xlsx, sheet_name='Milling Rate Data')
    fr = int_results['Frame']
    center_x = int_results['FOV X Center (Pix)']
    center_y = int_results['FOV Y Center (Pix)']
    sv_apert = np.min((51, len(fr)//8*2+1))
    trend_x = savgol_filter(center_x*1.0, sv_apert, 1) - center_x[0]
    trend_y = savgol_filter(center_y*1.0, sv_apert, 1) - center_y[0]

    if disp_res:
        print('Generating Plot')
    fs = 12

    fig, axs = plt.subplots(2,1, figsize = (6,7), sharex=True)
    fig.subplots_adjust(left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.05, hspace=0.05)
    axs[0].plot(fr, center_x, label='FOV X center, Data', color='red')
    axs[0].plot(fr, center_y, label='FOV Y center, Data', color='blue')
    axs[0].grid(True)
    axs[0].set_ylabel('FOV Center (Pix)')
    #axs[0].set_xlim(xi, xa)
    axs[0].legend(fontsize=12)

    axs[1].plot(fr, trend_x, label='FOV X center shift, smoothed', color='red')
    axs[1].plot(fr, trend_y, label='FOV Y center shift, smoothed', color='blue')
    axs[1].grid(True)
    axs[1].set_ylabel('FOV Center Shift (Pix)')
    axs[1].legend(fontsize=12)
    axs[1].set_xlabel('Frame')
    ldm = 70
    data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]
    try:
        axs[0].text(-0.15, 1.05, Sample_ID + '    ' +  data_dir_short, fontsize = fs-2, transform=axs[0].transAxes)
    except:
        axs[0].text(-0.15, 1.05, data_dir_short, fontsize = fs-2, transform=axs[0].transAxes)
    fig.savefig(os.path.join(data_dir, FOV_center_shift_xlsx.replace('.xlsx','_FOV_XYcenter.png')), dpi=300)
    return


def generate_report_data_minmax_xlsx(minmax_xlsx_file, **kwargs):
    '''
    Generate Report Plot for data Min-Max from XLSX spreadsheet file. ©G.Shtengel 10/2022 gleb.shtengel@gmail.com
    
    Parameters:
    minmax_xlsx_file : str
        Path to the XLSX spreadsheet file containing Min-Max data
    '''
    disp_res = kwargs.get('disp_res', False)
    if disp_res:
        print('Loading kwarg Data')
    saved_kwargs = read_kwargs_xlsx(minmax_xlsx_file, 'kwargs Info', **kwargs)
    data_dir = saved_kwargs.get("data_dir", '')
    fnm_reg = saved_kwargs.get("fnm_reg", 'Registration_file.mrc')
    Sample_ID = saved_kwargs.get("Sample_ID", '')
    thr_min = saved_kwargs.get("thr_min", 0.0)
    thr_max = saved_kwargs.get("thr_min", 0.0)
    fit_params_saved = saved_kwargs.get("fit_params", ['SG', 101, 3])
    fit_params = kwargs.get("fit_params", fit_params_saved)
    preserve_scales =  saved_kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    
    if disp_res:
        print('Loading MinMax Data')
    try:
        int_results = pd.read_excel(minmax_xlsx_file, sheet_name='FIBSEM Data')
    except:
        int_results = pd.read_excel(minmax_xlsx_file, sheet_name='MinMax Data')
    frames = int_results['Frame']
    frame_min = int_results['Min']
    frame_max = int_results['Max']
    data_min_glob  = np.min(frame_min)
    data_max_glob  = np.max(frame_max)
    '''
    sliding_min = int_results['Sliding Min']
    sliding_max = int_results['Sliding Max']
    '''
    if fit_params[0] != 'None':
        sv_apert = min([fit_params[1], len(frames)//8*2+1])
        print('Using fit_params: ', 'SG', sv_apert, fit_params[2])
        sliding_min = savgol_filter(frame_min.astype(np.double), sv_apert, fit_params[2])
        sliding_max = savgol_filter(frame_max.astype(np.double), sv_apert, fit_params[2])
    else:
        print('Not smoothing the Min/Max data')
        sliding_min = frame_min.astype(np.double)
        sliding_max = frame_min.astype(np.double)

    if disp_res:
        print('Generating Plot')
    fs = 12
    fig0, ax0 = plt.subplots(1,1,figsize=(6,4))
    fig0.subplots_adjust(left=0.14, bottom=0.11, right=0.99, top=0.94)
    ax0.plot(frame_min, 'b', linewidth=1, label='Frame Minima')
    ax0.plot(sliding_min, 'b', linewidth=2, linestyle = 'dotted', label='Sliding Minima')
    ax0.plot(frame_max, 'r', linewidth=1, label='Frame Maxima')
    ax0.plot(sliding_max, 'r', linewidth=2, linestyle = 'dotted', label='Sliding Maxima')
    ax0.legend()
    ax0.grid(True)
    ax0.set_xlabel('Frame')
    ax0.set_ylabel('Minima and Maxima Values')
    dxn = (data_max_glob - data_min_glob)*0.1
    ax0.set_ylim((data_min_glob - dxn, data_max_glob+dxn))
    # if needed, display the data in a narrower range
    #ax0.set_ylim((-4500, -1500))
    xminmax = [0, len(frame_min)]
    y_min = [data_min_glob, data_min_glob]
    y_max = [data_max_glob, data_max_glob]
    ax0.plot(xminmax, y_min, 'b', linestyle = '--')
    ax0.plot(xminmax, y_max, 'r', linestyle = '--')
    ax0.text(len(frame_min)/20.0, data_min_glob-dxn/1.75, 'data_min_glob={:.1f}'.format(data_min_glob), fontsize = fs-2, c='b')
    ax0.text(len(frame_min)/20.0, data_max_glob+dxn/2.25, 'data_max_glob={:.1f}'.format(data_max_glob), fontsize = fs-2, c='r')
    ax0.text(len(frame_min)/20.0, data_min_glob+dxn*4.5, 'thr_min={:.1e}'.format(thr_min), fontsize = fs-2, c='b')
    ax0.text(len(frame_min)/20.0, data_min_glob+dxn*5.5, 'thr_max={:.1e}'.format(thr_max), fontsize = fs-2, c='r')
    ldm = 70
    data_dir_short = data_dir if len(data_dir)<ldm else '... '+ data_dir[-ldm:]

    try:
        ax0.text(-0.15, 1.05, Sample_ID + '    ' +  data_dir_short, fontsize = fs-2, transform=axs[0].transAxes)
    except:
        ax0.text(-0.15, 1.05, data_dir_short, fontsize = fs-2, transform=ax0.transAxes)
    '''
    try:
        fig0.suptitle(Sample_ID + '    ' +  data_dir_short, fontsize = fs-2)
    except:
        fig0.suptitle(data_dir_short, fontsize = fs-2)
    '''
    fig0.savefig(os.path.join(data_dir, minmax_xlsx_file.replace('.xlsx','_Min_Max.png')), dpi=300)


def generate_report_transf_matrix_from_xlsx(transf_matrix_xlsx_file, **kwargs):
    '''
    Generate Report Plot for Transformation Matrix from XLSX spreadsheet file. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com
    
    Parameters:
    transf_matrix_xlsx_file : str
        Path to the XLSX spreadsheet file containing Transformation Matrix data

    '''
    disp_res = kwargs.get('disp_res', False)
    if disp_res:
        print('Loading kwarg Data')
    saved_kwargs = read_kwargs_xlsx(transf_matrix_xlsx_file, 'kwargs Info', **kwargs)
    data_dir = saved_kwargs.get("data_dir", '')
    fnm_reg = saved_kwargs.get("fnm_reg", 'Registration_file.mrc')
    TransformType = saved_kwargs.get("TransformType", RegularizedAffineTransform)
    Sample_ID = saved_kwargs.get("Sample_ID", '')
    SIFT_nfeatures = saved_kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = saved_kwargs.get("SIFT_nOctaveLayers", 3)
    SIFT_contrastThreshold = saved_kwargs.get("SIFT_contrastThreshold", 0.025)
    SIFT_edgeThreshold = saved_kwargs.get("SIFT_edgeThreshold", 10)
    SIFT_sigma = saved_kwargs.get("SIFT_sigma", 1.6)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = saved_kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = saved_kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = saved_kwargs.get("solver", 'RANSAC')
    drmax = saved_kwargs.get("drmax", 2.0)
    max_iter = saved_kwargs.get("max_iter", 1000)
    BFMatcher = saved_kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = saved_kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    save_res_png  = saved_kwargs.get("save_res_png", True)

    preserve_scales = saved_kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params = saved_kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit = saved_kwargs.get("subtract_linear_fit", [True, True])   # The linear slopes along X- and Y- directions (respectively) will be subtracted from the cumulative shifts.
    subtract_FOVtrend_from_fit = saved_kwargs.get("subtract_FOVtrend_from_fit", [True, True])
    pad_edges =  saved_kwargs.get("pad_edges", True)
    
    if disp_res:
        print('Loading Original Transformation Data')
    orig_transf_matrix = pd.read_excel(transf_matrix_xlsx_file, sheet_name='Orig. Transformation Matrix')
    transformation_matrix = np.vstack((orig_transf_matrix['T00 (Sxx)'],
                         orig_transf_matrix['T01 (Sxy)'],
                         orig_transf_matrix['T02 (Tx)'],
                         orig_transf_matrix['T10 (Syx)'],
                         orig_transf_matrix['T11 (Syy)'],
                         orig_transf_matrix['T12 (Ty)'],
                         orig_transf_matrix['T20 (0.0)'],
                         orig_transf_matrix['T21 (0.0)'],
                         orig_transf_matrix['T22 (1.0)'])).T.reshape((len(orig_transf_matrix['T00 (Sxx)']), 3, 3))
   
    if disp_res:
        print('Loading Cumulative Transformation Data')
    cum_transf_matrix = pd.read_excel(transf_matrix_xlsx_file, sheet_name='Cum. Transformation Matrix')
    tr_matr_cum = np.vstack((cum_transf_matrix['T00 (Sxx)'],
                         cum_transf_matrix['T01 (Sxy)'],
                         cum_transf_matrix['T02 (Tx)'],
                         cum_transf_matrix['T10 (Syx)'],
                         cum_transf_matrix['T11 (Syy)'],
                         cum_transf_matrix['T12 (Ty)'],
                         cum_transf_matrix['T20 (0.0)'],
                         cum_transf_matrix['T21 (0.0)'],
                         cum_transf_matrix['T22 (1.0)'])).T.reshape((len(cum_transf_matrix['T00 (Sxx)']), 3, 3))
    
    if disp_res:
        print('Loading Intermediate Data')

    int_results = pd.read_excel(transf_matrix_xlsx_file, sheet_name='Intermediate Results')
    s00_cum_orig = int_results['s00_cum_orig']
    s11_cum_orig = int_results['s11_cum_orig']
    s01_cum_orig = int_results['s01_cum_orig']
    s10_cum_orig = int_results['s10_cum_orig']
    Xshift_cum_orig = int_results['Xshift_cum_orig']
    Yshift_cum_orig = int_results['Yshift_cum_orig']
    
    if disp_res:
        print('Loading Statistics')
    stat_results = pd.read_excel(transf_matrix_xlsx_file, sheet_name='Reg. Stat. Info')
    npts = stat_results['Npts']
    error_abs_mean = stat_results['Mean Abs Error']
 
    fs = 14
    lwf = 2
    lwl = 1
    fig5, axs5 = plt.subplots(4,3, figsize=(18, 16), sharex=True)
    fig5.subplots_adjust(left=0.07, bottom=0.03, right=0.99, top=0.95)
    # display the info
    axs5[0,0].axis(False)
    axs5[0,0].text(-0.1, 0.9, Sample_ID, fontsize = fs + 4)
    #axs5[0,0].text(-0.1, 0.73, 'Global Data Range:  Min={:.2f}, Max={:.2f}'.format(data_min_glob, data_max_glob), transform=axs5[0,0].transAxes, fontsize = fs)
    
    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext = 'Reg.Aff.Transf., λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ' '.join(tstr) + '], w/' + solver
    else:
        otext = TransformType.__name__ + ' with ' + solver + ' solver'
    axs5[0,0].text(-0.1, 0.80, otext, transform=axs5[0,0].transAxes, fontsize = fs)

    SIFT1text = 'SIFT: nFeatures = {:d}, nOctaveLayers = {:d}, '.format(SIFT_nfeatures, SIFT_nOctaveLayers)
    axs5[0,0].text(-0.1, 0.65, SIFT1text, transform=axs5[0,0].transAxes, fontsize = fs)

    SIFT2text = 'SIFT: contrThr = {:.3f}, edgeThr = {:.2f}, σ= {:.2f}'.format(SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)
    axs5[0,0].text(-0.1, 0.50, SIFT2text, transform=axs5[0,0].transAxes, fontsize = fs)

    sbtrfit = ('ON, ' if  subtract_linear_fit[0] else 'OFF, ') + ('ON' if  subtract_linear_fit[1] else 'OFF') + ('(ON, ' if  subtract_FOVtrend_from_fit[0] else '(OFF, ') + ('ON)' if  subtract_FOVtrend_from_fit[1] else 'OFF)')
    axs5[0,0].text(-0.1, 0.35, 'drmax={:.1f}, Max # of Iter.={:d}'.format(drmax, max_iter), transform=axs5[0,0].transAxes, fontsize = fs)
    padedges = 'ON' if pad_edges else 'OFF'
    if preserve_scales:
        fit_method = fit_params[0]
        if fit_method == 'LF':
            fit_str = ', Meth: Linear Fit'
            fm_string = 'linear'
        else:
            if fit_method == 'SG':
                fit_str = ', Meth: Sav.-Gol., ' + str(fit_params[1:])
                fm_string = 'Sav.-Gol.'
            else:
                fit_str = ', Meth: Pol.Fit, ord.={:d}'.format(fit_params[1])
                fm_string = 'polyn.'
        preserve_scales_string = 'Pres. Scls: ON' + fit_str
    else:
        preserve_scales_string = 'Preserve Scales: OFF'
    axs5[0,0].text(-0.1, 0.20, preserve_scales_string, transform=axs5[0,0].transAxes, fontsize = fs)
    axs5[0,0].text(-0.1, 0.05, 'Subtract Shift Fit: ' + sbtrfit + ', Pad Edges: ' + padedges, transform=axs5[0,0].transAxes, fontsize = fs)
    # plot number of keypoints
    axs5[0, 1].plot(npts, 'g', linewidth = lwl, label = '# of key-points per frame')
    axs5[0, 1].set_title('# of key-points per frame')
    axs5[0, 1].text(0.03, 0.2, 'Mean # of kpts= {:.0f}   Median # of kpts= {:.0f}'.format(np.mean(npts), np.median(npts)), transform=axs5[0, 1].transAxes, fontsize = fs-1)
    # plot Standard deviations
    axs5[0, 2].plot(error_abs_mean, 'magenta', linewidth = lwl, label = 'Mean Abs Error over keyponts per frame')
    axs5[0, 2].set_title('Mean Abs Error keyponts per frame')  
    axs5[0, 2].text(0.03, 0.3, 'Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}'.format(np.mean(error_abs_mean), np.median(error_abs_mean)), transform=axs5[0, 2].transAxes, fontsize = fs-1)
    try:
        error_FWHMx = stat_results['Xerror_FWHM']
        axs5[0, 2].plot(error_FWHMx, 'red', linewidth = lwl, label = 'X-error FWHM')
        axs5[0, 2].text(0.03, 0.2, 'Mean X FWHM= {:.3f}   Median X FWHM= {:.3f}'.format(np.mean(error_FWHMx), np.median(error_FWHMx)), transform=axs5[0, 2].transAxes, fontsize = fs-1)
    except:
        print('No Xerror_FWHM data')
    try:
        error_FWHMy = stat_results['Yerror_FWHM']
        axs5[0, 2].plot(error_FWHMy, 'blue', linewidth = lwl, label = 'Y-error FWHM')
        axs5[0, 2].text(0.03, 0.1, 'Mean Y FWHM= {:.3f}   Median Y FWHM= {:.3f}'.format(np.mean(error_FWHMy), np.median(error_FWHMy)), transform=axs5[0, 2].transAxes, fontsize = fs-1)
    except:
        print('No Yerror_FWHM data')

    # plot scales terms
    axs5[1, 0].plot(transformation_matrix[:, 0, 0], 'r', linewidth = lwl, label = 'Sxx frame-to-frame')
    axs5[1, 0].plot(transformation_matrix[:, 1, 1], 'b', linewidth = lwl, label = 'Syy frame-to-frame')
    axs5[1, 0].set_title('Frame-to-Frame Scale Change', fontsize = fs)
    axs5[2, 0].plot(s00_cum_orig, 'r', linewidth = lwl, linestyle='dotted', label = 'Sxx cum.')
    axs5[2, 0].plot(s11_cum_orig, 'b', linewidth = lwl, linestyle='dotted', label = 'Syy cum.')
    if preserve_scales:
        try:
            s00_fit = int_results['s00_fit']
            s11_fit = int_results['s11_fit']
            axs5[2, 0].plot(s00_fit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Sxx cum. - '+fm_string+' fit')
            axs5[2, 0].plot(s11_fit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Syy cum. - '+fm_string+' fit')
        except:
            pass
    axs5[2, 0].set_title('Cumulative Scale', fontsize = fs)
    yi10,ya10 = axs5[1, 0].get_ylim()
    dy0 = (ya10-yi10)/2.0
    yi20,ya20 = axs5[2, 0].get_ylim()
    if (ya20-yi20)<0.01*dy0:
        axs5[2, 0].set_ylim((yi20-dy0, ya20+dy0))
    axs5[3, 0].plot(tr_matr_cum[:, 0, 0], 'r', linewidth = lwl, label = 'Sxx cum. - residual')
    axs5[3, 0].plot(tr_matr_cum[:, 1, 1], 'b', linewidth = lwl, label = 'Syy cum. - residual')
    axs5[3, 0].set_title('Residual Cumulative Scale', fontsize = fs)
    axs5[3, 0].set_xlabel('Frame', fontsize = fs+1)
    yi30,ya30 = axs5[3, 0].get_ylim()
    if (ya30-yi30)<0.01*dy0:
        axs5[3, 0].set_ylim((yi30-dy0, ya30+dy0))
    
    # plot shear terms
    axs5[1, 1].plot(transformation_matrix[:, 0, 1], 'r', linewidth = lwl, label = 'Sxy frame-to-frame')
    axs5[1, 1].plot(transformation_matrix[:, 1, 0], 'b', linewidth = lwl, label = 'Syx frame-to-frame')
    axs5[1, 1].set_title('Frame-to-Frame Shear Change', fontsize = fs)
    axs5[2, 1].plot(s01_cum_orig, 'r', linewidth = lwl, linestyle='dotted', label = 'Sxy cum.')
    axs5[2, 1].plot(s10_cum_orig, 'b', linewidth = lwl, linestyle='dotted', label = 'Syx cum.')
    if preserve_scales:
        try:
            s01_fit = int_results['s01_fit']
            s10_fit = int_results['s10_fit']
            axs5[2, 1].plot(s01_fit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Sxy cum. - '+fm_string+' fit')
            axs5[2, 1].plot(s10_fit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Syx cum. - '+fm_string+' fit')
        except:
            pass
    axs5[2, 1].set_title('Cumulative Shear', fontsize = fs)
    yi11,ya11 = axs5[1, 1].get_ylim()
    dy1 = (ya11-yi11)/2.0
    yi21,ya21 = axs5[2, 1].get_ylim()
    if (ya21-yi21)<0.01*dy1:
        axs5[2, 1].set_ylim((yi21-dy1, ya21+dy1))
    axs5[3, 1].plot(tr_matr_cum[:, 0, 1], 'r', linewidth = lwl, label = 'Sxy cum. - residual')
    axs5[3, 1].plot(tr_matr_cum[:, 1, 0], 'b', linewidth = lwl, label = 'Syx cum. - residual')
    axs5[3, 1].set_title('Residual Cumulative Shear', fontsize = fs)
    axs5[3, 1].set_xlabel('Frame', fontsize = fs+1)
    yi31,ya31 = axs5[3, 1].get_ylim()
    if (ya31-yi21)<0.01*dy1:
        axs5[3, 1].set_ylim((yi31-dy1, ya31+dy1))
    
    # plot shifts
    try:
        Xshift_cum = int_results['Xshift_cum']
    except:
        pass
    try:
        Yshift_cum = int_results['Yshift_cum']
    except:
        pass
    axs5[1, 2].plot(transformation_matrix[:, 0, 2], 'r', linewidth = lwl, label = 'Tx fr.-to-fr.')
    axs5[1, 2].plot(transformation_matrix[:, 1, 2], 'b', linewidth = lwl, label = 'Ty fr.-to-fr.')
    axs5[1, 2].set_title('Frame-to-Frame Shift', fontsize = fs)
    if preserve_scales:
        axs5[2, 2].plot(Xshift_cum_orig, 'r', linewidth = lwl, label = 'Tx cum. - orig.')
        axs5[2, 2].plot(Yshift_cum_orig, 'b', linewidth = lwl, label = 'Ty cum. - orig.')
        try:
            axs5[2, 2].plot(Xshift_cum, 'r', linewidth = lwl, linestyle='dotted', label = 'Tx cum. - pres. scales')
            axs5[2, 2].plot(Yshift_cum, 'b', linewidth = lwl, linestyle='dotted', label = 'Ty cum. - pres. scales')
        except:
            pass
    else:
        try:
            axs5[2, 2].plot(Xshift_cum, 'r', linewidth = lwl, linestyle='dotted', label = 'Tx cum.')
            axs5[2, 2].plot(Yshift_cum, 'b', linewidth = lwl, linestyle='dotted', label = 'Ty cum.')
        except:
            pass
    if subtract_linear_fit[0]:
        try:
            Xfit = int_results['Xfit']
            axs5[2, 2].plot(Xfit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Tx cum. - lin. fit')
        except:
            pass
    if subtract_linear_fit[1]:
        try:
            Yfit = int_results['Yfit']
            axs5[2, 2].plot(Yfit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Ty cum. - lin. fit')
        except:
            pass
    axs5[2, 2].set_title('Cumulative Shift', fontsize = fs)
    axs5[3, 2].plot(tr_matr_cum[:, 0, 2], 'r', linewidth = lwl, label = 'Tx cum. - residual')
    axs5[3, 2].plot(tr_matr_cum[:, 1, 2], 'b', linewidth = lwl, label = 'Ty cum. - residual')
    axs5[3, 2].set_title('Residual Cumulative Shift', fontsize = fs)
    axs5[3, 2].set_xlabel('Frame', fontsize = fs+1)

    for ax in axs5.ravel()[1:]:
        ax.grid(True)
        ax.legend(fontsize = fs-1)
    fig5.suptitle(transf_matrix_xlsx_file, fontsize = fs)
    if save_res_png :
        fig5.savefig(transf_matrix_xlsx_file.replace('.xlsx', '.png'), dpi=300)


def generate_report_transf_matrix_details(transf_matrix_bin_file, *kwarrgs):
    '''
    Generate Report Plot for Transformation Matrix from binary dump file. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com
    The binary dump file should contain list with these parameters (in this order):
        [saved_kwargs, npts, error_abs_mean, transformation_matrix,
        s00_cum_orig, s11_cum_orig, s00_fit, s11_fit, tr_matr_cum, s01_cum_orig, s10_cum_orig, s01_fit, s10_fit,
        Xshift_cum_orig, Yshift_cum_orig, Xshift_cum, Yshift_cum, Xfit, Yfit]

    Parameters:
    transf_matrix_bin_file : str
        Path to the binary dump file

    '''
    with open(transf_matrix_bin_file, "rb") as f:
        [saved_kwargs, npts, error_abs_mean,
         transformation_matrix, s00_cum_orig, s11_cum_orig, s00_fit, s11_fit,
         tr_matr_cum, s01_cum_orig, s10_cum_orig, s01_fit, s10_fit,
         Xshift_cum_orig, Yshift_cum_orig, Xshift_cum, Yshift_cum, Xfit, Yfit] = pickle.load(f)
    
    data_dir = saved_kwargs.get("data_dir", '')
    fnm_reg = saved_kwargs.get("fnm_reg", 'Registration_file.mrc')
    TransformType = saved_kwargs.get("TransformType", RegularizedAffineTransform)
    Sample_ID = saved_kwargs.get("Sample_ID", '')
    SIFT_nfeatures = saved_kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = saved_kwargs.get("SIFT_nOctaveLayers", 3)
    SIFT_contrastThreshold = saved_kwargs.get("SIFT_contrastThreshold", 0.025)
    SIFT_edgeThreshold = saved_kwargs.get("SIFT_edgeThreshold", 10)
    SIFT_sigma = saved_kwargs.get("SIFT_sigma", 1.6)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = saved_kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = saved_kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = saved_kwargs.get("solver", 'RANSAC')
    drmax = saved_kwargs.get("drmax", 2.0)
    max_iter = saved_kwargs.get("max_iter", 1000)
    BFMatcher = saved_kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = saved_kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    save_res_png  = saved_kwargs.get("save_res_png", True)

    preserve_scales =  saved_kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  saved_kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  saved_kwargs.get("subtract_linear_fit", [True, True])   # The linear slopes along X- and Y- directions (respectively) will be subtracted from the cumulative shifts.
    subtract_FOVtrend_from_fit = saved_kwargs.get("subtract_FOVtrend_from_fit", [True, True]) 
    #print("subtract_linear_fit:", subtract_linear_fit)
    pad_edges =  saved_kwargs.get("pad_edges", True)
    
    fs = 14
    lwf = 2
    lwl = 1
    fig5, axs5 = plt.subplots(4,3, figsize=(18, 16), sharex=True)
    fig5.subplots_adjust(left=0.07, bottom=0.03, right=0.99, top=0.95)
    # display the info
    axs5[0,0].axis(False)
    axs5[0,0].text(-0.1, 0.9, Sample_ID, fontsize = fs + 4)
    #axs5[0,0].text(-0.1, 0.73, 'Global Data Range:  Min={:.2f}, Max={:.2f}'.format(data_min_glob, data_max_glob), transform=axs5[0,0].transAxes, fontsize = fs)
    
    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext = 'Reg.Aff.Transf., λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ' '.join(tstr) + '], w/' + solver
    else:
        otext = TransformType.__name__ + ' with ' + solver + ' solver'
    axs5[0,0].text(-0.1, 0.80, otext, transform=axs5[0,0].transAxes, fontsize = fs)

    SIFT1text = 'SIFT: nFeatures = {:d}, nOctaveLayers = {:d}, '.format(SIFT_nfeatures, SIFT_nOctaveLayers)
    axs5[0,0].text(-0.1, 0.65, SIFT1text, transform=axs5[0,0].transAxes, fontsize = fs)

    SIFT2text = 'SIFT: contrThr = {:.3f}, edgeThr = {:.2f}, σ= {:.2f}'.format(SIFT_contrastThreshold, SIFT_edgeThreshold, SIFT_sigma)
    axs5[0,0].text(-0.1, 0.50, SIFT2text, transform=axs5[0,0].transAxes, fontsize = fs)

    sbtrfit = ('ON, ' if  subtract_linear_fit[0] else 'OFF, ') + ('ON' if  subtract_linear_fit[1] else 'OFF')
    axs5[0,0].text(-0.1, 0.35, 'drmax={:.1f}, Max # of Iter.={:d}'.format(drmax, max_iter), transform=axs5[0,0].transAxes, fontsize = fs)
    padedges = 'ON' if pad_edges else 'OFF'
    if preserve_scales:
        fit_method = fit_params[0]
        if fit_method == 'LF':
            fit_str = ', Meth: Linear Fit'
            fm_string = 'linear'
        else:
            if fit_method == 'SG':
                fit_str = ', Meth: Sav.-Gol., ' + str(fit_params[1:])
                fm_string = 'Sav.-Gol.'
            else:
                fit_str = ', Meth: Pol.Fit, ord.={:d}'.format(fit_params[1])
                fm_string = 'polyn.'
        preserve_scales_string = 'Pres. Scls: ON' + fit_str
    else:
        preserve_scales_string = 'Preserve Scales: OFF'
    axs5[0,0].text(-0.1, 0.20, preserve_scales_string, transform=axs5[0,0].transAxes, fontsize = fs)
    axs5[0,0].text(-0.1, 0.05, 'Subtract Shift Fit: ' + sbtrfit + ', Pad Edges: ' + padedges, transform=axs5[0,0].transAxes, fontsize = fs)
    # plot number of keypoints
    axs5[0, 1].plot(npts, 'g', linewidth = lwl, label = '# of key-points per frame')
    axs5[0, 1].set_title('# of key-points per frame')
    axs5[0, 1].text(0.03, 0.2, 'Mean # of kpts= {:.0f}   Median # of kpts= {:.0f}'.format(np.mean(npts), np.median(npts)), transform=axs5[0, 1].transAxes, fontsize = fs-1)
    # plot Standard deviations
    axs5[0, 2].plot(error_abs_mean, 'magenta', linewidth = lwl, label = 'Mean Abs Error over keyponts per frame')
    axs5[0, 2].set_title('Mean Abs Error keyponts per frame')  
    axs5[0, 2].text(0.03, 0.2, 'Mean Abs Error= {:.3f}   Median Abs Error= {:.3f}'.format(np.mean(error_abs_mean), np.median(error_abs_mean)), transform=axs5[0, 2].transAxes, fontsize = fs-1)
    
    # plot scales terms
    axs5[1, 0].plot(transformation_matrix[:, 0, 0], 'r', linewidth = lwl, label = 'Sxx frame-to-frame')
    axs5[1, 0].plot(transformation_matrix[:, 1, 1], 'b', linewidth = lwl, label = 'Syy frame-to-frame')
    axs5[1, 0].set_title('Frame-to-Frame Scale Change', fontsize = fs)
    axs5[2, 0].plot(s00_cum_orig, 'r', linewidth = lwl, linestyle='dotted', label = 'Sxx cum.')
    axs5[2, 0].plot(s11_cum_orig, 'b', linewidth = lwl, linestyle='dotted', label = 'Syy cum.')
    if preserve_scales:
        axs5[2, 0].plot(s00_fit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Sxx cum. - '+fm_string+' fit')
        axs5[2, 0].plot(s11_fit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Syy cum. - '+fm_string+' fit')
    axs5[2, 0].set_title('Cumulative Scale', fontsize = fs)
    yi10,ya10 = axs5[1, 0].get_ylim()
    dy0 = (ya10-yi10)/2.0
    yi20,ya20 = axs5[2, 0].get_ylim()
    if (ya20-yi20)<0.01*dy0:
        axs5[2, 0].set_ylim((yi20-dy0, ya20+dy0))
    axs5[3, 0].plot(tr_matr_cum[:, 0, 0], 'r', linewidth = lwl, label = 'Sxx cum. - residual')
    axs5[3, 0].plot(tr_matr_cum[:, 1, 1], 'b', linewidth = lwl, label = 'Syy cum. - residual')
    axs5[3, 0].set_title('Residual Cumulative Scale', fontsize = fs)
    axs5[3, 0].set_xlabel('Frame', fontsize = fs+1)
    yi30,ya30 = axs5[3, 0].get_ylim()
    if (ya30-yi30)<0.01*dy0:
        axs5[3, 0].set_ylim((yi30-dy0, ya30+dy0))
    
    # plot shear terms
    axs5[1, 1].plot(transformation_matrix[:, 0, 1], 'r', linewidth = lwl, label = 'Sxy frame-to-frame')
    axs5[1, 1].plot(transformation_matrix[:, 1, 0], 'b', linewidth = lwl, label = 'Syx frame-to-frame')
    axs5[1, 1].set_title('Frame-to-Frame Shear Change', fontsize = fs)
    axs5[2, 1].plot(s01_cum_orig, 'r', linewidth = lwl, linestyle='dotted', label = 'Sxy cum.')
    axs5[2, 1].plot(s10_cum_orig, 'b', linewidth = lwl, linestyle='dotted', label = 'Syx cum.')
    if preserve_scales:
        axs5[2, 1].plot(s01_fit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Sxy cum. - '+fm_string+' fit')
        axs5[2, 1].plot(s10_fit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Syx cum. - '+fm_string+' fit')
    axs5[2, 1].set_title('Cumulative Shear', fontsize = fs)
    yi11,ya11 = axs5[1, 1].get_ylim()
    dy1 = (ya11-yi11)/2.0
    yi21,ya21 = axs5[2, 1].get_ylim()
    if (ya21-yi21)<0.01*dy1:
        axs5[2, 1].set_ylim((yi21-dy1, ya21+dy1))
    axs5[3, 1].plot(tr_matr_cum[:, 0, 1], 'r', linewidth = lwl, label = 'Sxy cum. - residual')
    axs5[3, 1].plot(tr_matr_cum[:, 1, 0], 'b', linewidth = lwl, label = 'Syx cum. - residual')
    axs5[3, 1].set_title('Residual Cumulative Shear', fontsize = fs)
    axs5[3, 1].set_xlabel('Frame', fontsize = fs+1)
    yi31,ya31 = axs5[3, 1].get_ylim()
    if (ya31-yi21)<0.01*dy1:
        axs5[3, 1].set_ylim((yi31-dy1, ya31+dy1))
    
    # plot shifts
    axs5[1, 2].plot(transformation_matrix[:, 0, 2], 'r', linewidth = lwl, label = 'Tx fr.-to-fr.')
    axs5[1, 2].plot(transformation_matrix[:, 1, 2], 'b', linewidth = lwl, label = 'Ty fr.-to-fr.')
    axs5[1, 2].set_title('Frame-to-Frame Shift', fontsize = fs)
    if preserve_scales:
        axs5[2, 2].plot(Xshift_cum_orig, 'r', linewidth = lwl, label = 'Tx cum. - orig.')
        axs5[2, 2].plot(Yshift_cum_orig, 'b', linewidth = lwl, label = 'Ty cum. - orig.')
        axs5[2, 2].plot(Xshift_cum, 'r', linewidth = lwl, linestyle='dotted', label = 'Tx cum. - pres. scales')
        axs5[2, 2].plot(Yshift_cum, 'b', linewidth = lwl, linestyle='dotted', label = 'Ty cum. - pres. scales')
    else:
        axs5[2, 2].plot(Xshift_cum, 'r', linewidth = lwl, linestyle='dotted', label = 'Tx cum.')
        axs5[2, 2].plot(Yshift_cum, 'b', linewidth = lwl, linestyle='dotted', label = 'Ty cum.')
    if subtract_linear_fit[0]:
        axs5[2, 2].plot(Xfit, 'orange', linewidth = lwf, linestyle='dashed', label = 'Tx cum. - lin. fit')
    if subtract_linear_fit[1]:
        axs5[2, 2].plot(Yfit, 'cyan', linewidth = lwf, linestyle='dashed', label = 'Ty cum. - lin. fit')
    axs5[2, 2].set_title('Cumulative Shift', fontsize = fs)
    axs5[3, 2].plot(tr_matr_cum[:, 0, 2], 'r', linewidth = lwl, label = 'Tx cum. - residual')
    axs5[3, 2].plot(tr_matr_cum[:, 1, 2], 'b', linewidth = lwl, label = 'Ty cum. - residual')
    axs5[3, 2].set_title('Residual Cumulative Shift', fontsize = fs)
    axs5[3, 2].set_xlabel('Frame', fontsize = fs+1)

    for ax in axs5.ravel()[1:]:
        ax.grid(True)
        ax.legend(fontsize = fs-1)
    fn = os.path.join(data_dir, fnm_reg)
    fig5.suptitle(fn, fontsize = fs)
    if save_res_png :
        fig5.savefig(fn.replace('.mrc', '_Transform_Summary.png'), dpi=300)


def generate_report_from_xls_registration_summary(file_xlsx, **kwargs):
    '''
    Generate Report Plot for FIB-SEM data set registration from xlxs workbook file. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com
    XLS file should have pages (sheets):
        - 'Registration Quality Statistics' - containing columns with the the evaluation box data and registration quality metrics data: 
            'Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval', 'Npts', 'Mean Abs Error', 'NSAD', 'NCC', 'NMI'
        - 'Stack Info' - containing the fields:
            'Stack Filename' and 'data_dir'
        - 'SIFT kwargs' (optional) - containg the kwargs with SIFT registration parameters.

    Parameters:
    xlsx_fname : str
        full path to the XLSX spreadsheet file
    
    kwargs:
    ---------
    sample_frame_files : list
        List of paths to sample frame images
    png_file : str
        filename to save the results. Default is file_xlsx with extension '.xlsx' replaced with '.png'
    invert_data : boolean
        If True, the representative data frames will use inverse LUT. 
    sample_frames_layout : str
        Default is 'vertical'. Another option is 'horizontal'.
    full_filename_path : boolean
        Default is True. If False, only the filename, no path is printed in the title
    dump_filename : str
        Filename of a binary dump of the FIBSEM_dataset object.

    '''
    xlsx_name = os.path.basename(os.path.abspath(file_xlsx))
    base_dir = os.path.dirname(os.path.abspath(file_xlsx))
    sample_frame_mask = xlsx_name.replace('_RegistrationQuality.xlsx', '_sample_image_frame*.*')
    unsorted_sample_frame_files = glob.glob(os.path.join(base_dir, sample_frame_mask))
    try:
        unsorted_frames = [int(x.split('frame')[1].split('.png')[0]) for x in unsorted_sample_frame_files]
        sorted_inds = np.argsort(unsorted_frames)
        existing_sample_frame_files = [unsorted_sample_frame_files[i] for i in sorted_inds]
    except:
        existing_sample_frame_files = unsorted_sample_frame_files
    sample_frame_files = kwargs.get('sample_frame_files', existing_sample_frame_files)
    png_file_default = file_xlsx.replace('.xlsx','.png')
    png_file = kwargs.get("png_file", png_file_default)
    dump_filename = kwargs.get("dump_filename", '')
    sample_frames_layout = kwargs.get('sample_frames_layout', 'vertical')
    full_filename_path = kwargs.get('full_filename_path', True)
    
    Regisration_data = pd.read_excel(file_xlsx, sheet_name='Registration Quality Statistics')
    # columns=['Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval', 'Npts', 'Mean Abs Error', 'NSAD', 'NCC', 'NMI']
    frames = Regisration_data['Frame']
    xi_evals = Regisration_data['xi_eval']
    xa_evals = Regisration_data['xa_eval']
    yi_evals = Regisration_data['yi_eval']
    ya_evals = Regisration_data['ya_eval']
    
    '''
    image_nsad = Regisration_data['NSAD']
    image_ncc = Regisration_data['NCC']
    image_nmi = Regisration_data['NMI']
    nsads = [np.mean(image_nsad), np.median(image_nsad), np.std(image_nsad)] 
    nccs = [np.mean(image_ncc), np.median(image_ncc), np.std(image_ncc)]
    nmis = [np.mean(image_nmi), np.median(image_nmi), np.std(image_nmi)]
    '''
    eval_metrics = Regisration_data.columns[5:]
    num_metrics = len(eval_metrics)
    
    num_frames = len(frames)
    
    stack_info_dict = read_kwargs_xlsx(file_xlsx, 'Stack Info', **kwargs)
    if 'dump_filename' in stack_info_dict.keys():
        dump_filename = kwargs.get("dump_filename", stack_info_dict['dump_filename'])
    else:
        dump_filename = kwargs.get("dump_filename", '')
    try:
        if np.isnan(dump_filename):
            dump_filename = ''
    except:
        pass
    stack_info_dict['dump_filename'] = dump_filename
    
    try:
        invert_data =  kwargs.get("invert_data", stack_info_dict['invert_data'])
    except:
        invert_data =  kwargs.get("invert_data", False)

    default_stack_name = file_xlsx.replace('_RegistrationQuality.xlsx','.mrc')
    stack_filename = os.path.normpath(stack_info_dict.get('Stack Filename', default_stack_name))
    data_dir = stack_info_dict.get('data_dir', '')
    ftype = stack_info_dict.get("ftype", 0)
    
    fig_xsize = 6.0
    if sample_frames_layout == 'horizontal':
        fig_ysize = fig_xsize/3.0*(num_metrics+1)
        heights = [1.0]*(num_metrics+1)
        fig = plt.figure(figsize=(fig_xsize, fig_ysize))
        fig.subplots_adjust(left=0.14, bottom=0.04, right=0.99, top=0.98, wspace=0.01, hspace=0.04)
        gs = GridSpec(num_metrics+1, 3, figure=fig)#, height_ratios=heights)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax02 = fig.add_subplot(gs[0, 2])
        axs = [ax00, ax01, ax02]
        for j, metric in enumerate(eval_metrics):
            axs.append(fig.add_subplot(gs[j+1, :]))
    else:
        fig_ysize = fig_xsize/3.0*(num_metrics+2)
        heights = [0.8]*3 + [1.5]*num_metrics        
        fig = plt.figure(figsize=(fig_xsize, fig_ysize))
        fig.subplots_adjust(left=0.14, bottom=0.04, right=0.99, top=0.98, wspace=0.18, hspace=0.04)
        gs = GridSpec(num_metrics+3, 1, figure=fig, height_ratios=heights)
        axs = [fig.add_subplot(gss) for gss in gs]
    for ax in axs[0:3]:
        ax.axis(False)
    
    fs=12
    lwl=1
    
    if len(sample_frame_files)>0:
        sample_frame_images_available = True
        for jf, ax in enumerate(axs[0:3]):
            try:
                ax.imshow(mpimg.imread(sample_frame_files[jf]))
            except:
                pass
    else:
        sample_frame_images_available = False
        sample_data_available = True
        if stack_exists:
            print('Will use sample images from the registered stack')
            use_raw_data = False
            if Path(stack_filename).suffix == '.mrc':
                mrc_obj = mrcfile.mmap(stack_filename, mode='r')
                header = mrc_obj.header 
                mrc_mode = header.mode
                '''
                mode 0 -> uint8
                mode 1 -> int16
                mode 2 -> float32
                mode 4 -> complex64
                mode 6 -> uint16
                '''
                if mrc_mode==0:
                    dt_mrc=np.uint8
                if mrc_mode==1:
                    dt_mrc=np.int16
                if mrc_mode==2:
                    dt_mrc=np.float32
                if mrc_mode==4:
                    dt_mrc=np.complex64
                if mrc_mode==6:
                    dt_mrc=np.uint16
        else:
            print('Will use sample images from the raw data')
            if os.path.exists(dump_filename):
                print('Trying to recall the data from ', dump_filename)
            try:
                print('Looking for the raw data in the directory', data_dir)
                if ftype == 0:
                    fls = sorted(glob.glob(os.path.join(data_dir,'*.dat')))
                    if len(fls) < 1:
                        fls = sorted(glob.glob(os.path.join(data_dir,'*/*.dat')))
                if ftype == 1:
                    fls = sorted(glob.glob(os.path.join(data_dir,'*.tif')))
                    if len(fls) < 1:
                        fls = sorted(glob.glob(os.path.join(data_dir,'*/*.tif')))
                num_frames = len(fls)
                stack_info_dict['disp_res']=False
                raw_dataset = FIBSEM_dataset(fls, recall_parameters=os.path.exists(dump_filename), **stack_info_dict)
                XResolution = raw_dataset.XResolution
                YResolution = raw_dataset.YResolution
                shape = [YResolution, XResolution]
                if pad_edges and perform_transformation:
                    shapes = (raw_dataset.YResolutions, raw_dataset.XResolutions)
                    xi, yi, xsz, ysz = determine_sizes_and_offsets(shapes, raw_dataset.tr_matr_cum_residual)
                    # xi, yi, padx, pady = determine_pad_offsets(shape, raw_dataset.tr_matr_cum_residual)
                    # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
                    # so that the transformed images are not clipped.
                    # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
                    # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
                    # those are calculated below base on the amount of padding calculated above
                    shift_matrix = np.array([[1.0, 0.0, xi],
                                             [0.0, 1.0, yi],
                                             [0.0, 0.0, 1.0]]).astype(float)
                    inv_shift_matrix = np.linalg.inv(shift_matrix)
                else:
                    xsz = XResolution
                    ysz = YResolution
                    xi = 0
                    yi = 0
                    shift_matrix = np.eye(3,3)
                    inv_shift_matrix = np.eye(3,3)
                xa = xi + XResolution
                ya = yi + YResolution
                use_raw_data = True
            except:
                sample_data_available = False
                use_raw_data = False
        if sample_data_available:
            print('Sample data is available')
        else:
            print('Sample data is NOT available')
     
        if num_frames//10*9 > 0:
            ev_ind2 = num_frames//10*9
        else:
            ev_ind2 = num_frames-1
        eval_inds = [num_frames//10,  num_frames//2, ev_ind2]
        #print(eval_inds)

        for j, eval_ind in enumerate(eval_inds):
            ax = axs[j]
            if sample_data_available:
                if stack_exists:
                    if Path(stack_filename).suffix == '.mrc':
                        frame_img = (mrc_obj.data[frames[eval_ind], :, :].astype(dt_mrc)).astype(float)
                    if Path(stack_filename).suffix == '.tif':
                        frame_img = tiff.imread(stack_filename, key=eval_ind)
                else:
                    dtp=float
                    chunk_frames = np.arange(eval_ind, min(eval_ind+zbin_factor, len(fls)-2))
                    frame_filenames = np.array(raw_dataset.fls)[chunk_frames]
                    tr_matrices = np.array(raw_dataset.tr_matr_cum_residual)[chunk_frames]
                    frame_img = transform_chunk_of_frames(frame_filenames, xsz, ysz, ftype,
                            flatten_image, image_correction_file,
                            perform_transformation, tr_matrices, shift_matrix, inv_shift_matrix,
                            xi, xa, yi, ya,
                            ImgB_fraction=0.0,
                            invert_data=False,
                            int_order=1,
                            flipY = raw_dataset.flipY)
                #print(eval_ind, np.shape(frame_img), yi_evals[eval_ind], ya_evals[eval_ind], xi_evals[eval_ind], xa_evals[eval_ind])
                if use_raw_data:
                    eval_ind = eval_ind//zbin_factor
                dmin, dmax = get_min_max_thresholds(frame_img[yi_evals[eval_ind]:ya_evals[eval_ind], xi_evals[eval_ind]:xa_evals[eval_ind]], disp_res=False)
                if invert_data:
                    ax.imshow(frame_img, cmap='Greys_r', vmin=dmin, vmax=dmax)
                else:
                    ax.imshow(frame_img, cmap='Greys', vmin=dmin, vmax=dmax)

                #ax.text(0.03, 1.01, 'Frame={:d},  NSAD={:.3f},  NCC={:.3f},  NMI={:.3f}'.format(frames[eval_ind], image_nsad[eval_ind], image_ncc[eval_ind], image_nmi[eval_ind]), color='red', transform=ax.transAxes)
                ax.text(0.02, 0.95, 'Frame={:d}'.format(frames[eval_ind]), color='red', transform=ax.transAxes, fontsize=11)
                ax.text(0.50, 0.95, 'NSAD={:.3f}'.format(image_nsad[eval_ind]), color='red', transform=ax.transAxes, fontsize=11)
                ax.text(0.50, 0.85, 'NCC={:.3f}'.format(image_ncc[eval_ind]), color='red', transform=ax.transAxes, fontsize=11)
                ax.text(0.50, 0.75, 'NMI={:.3f}'.format(image_mi[eval_ind]), color='red', transform=ax.transAxes, fontsize=11)
                rect_patch = patches.Rectangle((xi_evals[eval_ind], yi_evals[eval_ind]), np.abs(xa_evals[eval_ind]-xi_evals[eval_ind])-2, np.abs(ya_evals[eval_ind]-yi_evals[eval_ind])-2, linewidth=1.0, edgecolor='yellow',facecolor='none')
                ax.add_patch(rect_patch)
            ax.axis('off')

        if stack_exists:
            if Path(stack_filename).suffix == '.mrc':
                mrc_obj.close()
    
    axes_names = {'NSAD' : 'Norm. Sum of Abs. Diff',
                 'NCC' : 'Norm. Cross-Corr.',
                  'NMI' : 'Norm. Mutual Inf.',
                 'FSC' : 'FSC BW (inv pix)'}
    colors = ['red', 'blue', 'green', 'magenta', 'lime']
    
    for j, metric in enumerate(eval_metrics):
        metric_data = Regisration_data[metric]
        nmis = []
        axs[j+3].plot(frames, Regisration_data[metric], linewidth=lwl, color = colors[j])
        try:
            axs[j+3].set_ylabel(axes_names[metric], fontsize=fs-2)
        except:
            axs[j+3].set_ylabel(metric, fontsize=fs-2)
        axs[j+3].text(0.02, 0.04, (metric+' mean = {:.3f}   ' + metric + ' median = {:.3f}  ' + metric + ' STD = {:.3f}').format(np.mean(metric_data), np.median(metric_data), np.std(metric_data)), transform=axs[j+3].transAxes, fontsize = fs-4)
        
    axs[-1].set_xlabel('Binned Frame #')
    for ax in axs[2:]:
        ax.grid(True)
    if full_filename_path:
        ttl = stack_filename
    else:
        ttl = os.path.split(stack_filename)[-1]
    if sample_frames_layout == 'horizontal':
        axs[0].text(-0.15, 1.05, ttl, transform=axs[0].transAxes)
    else:
        axs[0].set_title(ttl)
    fig.savefig(png_file, dpi=300)


def plot_registrtion_quality_xlsx(data_files, labels, **kwargs):
    '''
    Read and plot together multiple registration quality summaries.
    ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

    Parameters:
    data_files : array of str
        Filenames (full paths) of the registration summaries (*.xlsx files)
    labels : array of str
        Labels (for each registration)

    kwargs:
    frame_inds : array or list of int
        Array or list of frame indices to use to azalyze the data.
    save_res_png : boolean
        If True, the PNG's of summary plots as well as summary Excel notebook are saved.
    save_filename : str
        Filename (full path) to save the results (default is data_dir +'Regstration_Summary.png').
    nsad_bounds : list of floats
        Bounds for NSAD plot (default is determined by get_min_max_thresholds with thresholds of 1e-4).
    ncc_bounds : list of floats
        Bounds for NCC plot (default is determined by get_min_max_thresholds with thresholds of 1e-4).
    nmi_bounds : list of floats
        Bounds for NMI plot (default is determined by get_min_max_thresholds with thresholds of 1e-4).
    colors : array or list of colors
        Optional colors for each plot/file. If not provided, will be auto-generated.
    linewidths : array of float
        linewidths for individual files. If not provided, all linewidths are set to 0.5

    Returns
    xlsx_fname : str
        Filename of the summary Excel notebook.
    '''
    save_res_png  = kwargs.get("save_res_png", True )
    linewidths = kwargs.get("linewidths", np.ones(len(data_files))*0.5)
    data_dir = os.path.split(data_files[0])[0]
    default_save_filename = os.path.join(data_dir, 'Regstration_Summary.png')
    save_filename = kwargs.get("save_filename", default_save_filename)
    nsad_bounds = kwargs.get("nsad_bounds", [0.0, 0.0])
    ncc_bounds = kwargs.get("ncc_bounds", [0.0, 0.0])
    nmi_bounds = kwargs.get("nmi_bounds", [0.0, 0.0])
    frame_inds = kwargs.get("frame_inds", [])

    nfls = len(data_files)
    reg_datas = []
    for data_file in data_files:
        # fl = os.path.join(data_dir, df)
        # data = pd.read_csv(fl)
        data = pd.read_excel(data_file, sheet_name='Registration Quality Statistics')
        reg_datas.append(data)

    lw0 = 0.5
    lw1 = 1
    
    fs=12
    fs2=10
    fig1, axs1 = plt.subplots(3,1, figsize=(7, 11), sharex=True)
    fig1.subplots_adjust(left=0.1, bottom=0.05, right=0.99, top=0.96, wspace=0.2, hspace=0.1)

    ax_nsad = axs1[0]
    ax_ncc = axs1[1]
    ax_nmi = axs1[2]
    ax_nsad.set_ylabel('Normalized Sum of Abs. Differences', fontsize=fs)
    ax_ncc.set_ylabel('Normalized Cross-Correlation', fontsize=fs)
    ax_nmi.set_ylabel('Normalized Mutual Information', fontsize=fs)
    ax_nmi.set_xlabel('Frame', fontsize=fs)

    spreads=[]
    my_cols = [plt.get_cmap("gist_rainbow_r")((nfls-j)/(nfls)) for j in np.arange(nfls)]
    my_cols[0] = 'grey'
    my_cols[-1] = 'red'
    my_cols = kwargs.get("colors", my_cols)

    means = []
    image_nsads= []
    image_nccs= []
    image_snrs= []
    image_nmis = []
    frame_inds_glob = []
    for j, reg_data in enumerate(tqdm(reg_datas, desc='generating the registration quality summary plots')):
        #my_col = plt.get_cmap("gist_rainbow_r")((nfls-j)/(nfls))
        #my_cols.append(my_col)
        my_col = my_cols[j]
        pf = labels[j]
        lw0 = linewidths[j]
        if len(frame_inds)>0:
            try:
                image_nsad = np.array(reg_data['Image NSAD'])[frame_inds]
                image_ncc = np.array(reg_data['Image NCC'])[frame_inds]
                image_nmi = np.array(reg_data['Image MI'])[frame_inds]
            except:
                image_nsad = np.array(reg_data['NSAD'])[frame_inds]
                image_ncc = np.array(reg_data['NCC'])[frame_inds]
                image_nmi = np.array(reg_data['NMI'])[frame_inds]
            frame_inds_loc = frame_inds.copy()
        else:
            try:
                image_nsad = np.array(reg_data['Image NSAD'])
                image_ncc = np.array(reg_data['Image NCC'])
                image_nmi = np.array(reg_data['Image MI'])
            except:
                image_nsad = np.array(reg_data['NSAD'])
                image_ncc = np.array(reg_data['NCC'])
                image_nmi = np.array(reg_data['NMI'])
            frame_inds_loc = np.arange(len(image_ncc))
        fr_i = min(frame_inds_loc) - (max(frame_inds_loc) - min(frame_inds_loc))*0.05
        fr_a = max(frame_inds_loc) + (max(frame_inds_loc) - min(frame_inds_loc))*0.05
        image_nsads.append(image_nsad)
        image_nccs.append(image_ncc)
        image_snr = image_ncc/(1.0-image_ncc)
        image_snrs.append(image_snr)
        image_nmis.append(image_nmi)
        frame_inds_glob.append(frame_inds_loc)

        eval_metrics = [image_nsad, image_ncc, image_snr, image_nmi]
        spreads.append([get_spread(metr) for metr in eval_metrics])
        means.append([np.mean(metr) for metr in eval_metrics])

        ax_nsad.plot(frame_inds_loc, image_nsad, c=my_col, linewidth=lw0)
        ax_nsad.plot(image_nsad[0], c=my_col, linewidth=lw1, label=pf)
        ax_ncc.plot(frame_inds_loc, image_ncc, c=my_col, linewidth=lw0)
        ax_ncc.plot(image_ncc[0], c=my_col, linewidth=lw1, label=pf)
        ax_nmi.plot(frame_inds_loc, image_nmi, c=my_col, linewidth=lw0)
        ax_nmi.plot(image_nmi[0], c=my_col, linewidth=lw1, label=pf)

    for ax in axs1.ravel():
        ax.grid(True)
        ax.legend(fontsize=fs2)
        
    if nsad_bounds[0]==nsad_bounds[1]:
        nsad_min, nsad_max = get_min_max_thresholds(np.concatenate(image_nsads),
                                                    thr_min=1e-4, thr_max=1e-4,
                                                    nbins=256, disp_res=False)
    else:
        nsad_min, nsad_max = nsad_bounds
    ax_nsad.set_ylim(nsad_min, nsad_max)

    if ncc_bounds[0]==ncc_bounds[1]:
        ncc_min, ncc_max = get_min_max_thresholds(np.concatenate(image_nccs),
                                                    thr_min=1e-4, thr_max=1e-4,
                                                    nbins=256, disp_res=False)
    else:
        ncc_min, ncc_max = ncc_bounds
    ax_ncc.set_ylim(ncc_min, ncc_max)

    if nmi_bounds[0]==nmi_bounds[1]:
        nmi_min, nmi_max = get_min_max_thresholds(np.concatenate(image_nmis),
                                                    thr_min=1e-4, thr_max=1e-4,
                                                    nbins=256, disp_res=False)
    else:
        nmi_min, nmi_max = nmi_bounds
    ax_nmi.set_ylim(nmi_min, nmi_max)
    ax_nmi.set_xlim(fr_i, fr_a)

    
    ax_nsad.text(-0.05, 1.05, data_dir, transform=ax_nsad.transAxes, fontsize=10)
    if save_res_png:
        fig1.savefig(save_filename, dpi=300)
        
    # Generate the Cell Text
    cell_text = []
    fig2_data = []
    limits = []
    rows = labels
    fst=9

    for j, (mean, spread) in enumerate(zip(means, spreads)):
        cell_text.append(['{:.4f}'.format(mean[0]), '{:.4f}'.format(spread[0]),
                          '{:.4f}'.format(mean[1]), '{:.4f}'.format(spread[1]), '{:.4f}'.format(mean[2]),
                          '{:.4f}'.format(mean[3]), '{:.4f}'.format(spread[3])])
        fig2_data.append([mean[0], spread[0], mean[1], spread[1], mean[2], mean[3], spread[3]])
        
    # Generate the table
    fig2, ax = plt.subplots(1, 1, figsize=(9.5,1.3))
    fig2.subplots_adjust(left=0.32, bottom=0.01, right=0.98, top=0.86, wspace=0.05, hspace=0.05)
    ax.axis(False)
    ax.text(-0.30, 1.07, 'SIFT Registration Comparisons:  ' + data_dir, fontsize=fst)
    llw1=0.3
    clw = [llw1, llw1]

    columns = ['NSAD Mean', 'NSAD Spread', 'NCC Mean', 'NCC Spread', 'Mean SNR', 'NMI Mean', 'NMI Spread']

    n_cols = len(columns)
    n_rows = len(rows)

    tbl = ax.table(cellText = cell_text,
                   rowLabels = rows,
                   colLabels = columns,
                   cellLoc = 'center',
                   colLoc = 'center',
                   bbox = [0.01, 0, 0.995, 1.0],
                  fontsize=16)
    tbl.auto_set_column_width(col=3)

    table_props = tbl.properties()
    try:
        table_cells = table_props['child_artists']
    except:
        table_cells = table_props['children']

    tbl.auto_set_font_size(False)
    for j, cell in enumerate(table_cells[0:n_cols*n_rows]):
        cell.get_text().set_color(my_cols[j//n_cols])
        cell.get_text().set_fontsize(fst)
    for j, cell in enumerate(table_cells[n_cols*(n_rows+1):]):
        cell.get_text().set_color(my_cols[j])
    for cell in table_cells[n_cols*n_rows:]:
    #    cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(fst)
    save_filename2 = save_filename.replace('.png', '_table.png')
    if save_res_png:
        fig2.savefig(save_filename2, dpi=300)   
        
    ysize_fig = 4
    ysize_tbl = 0.25 * nfls
    fst3 = 6
    fig3, axs3 = plt.subplots(2, 1, figsize=(7, ysize_fig+ysize_tbl),  gridspec_kw={"height_ratios" : [ysize_tbl, ysize_fig]})
    fig3.subplots_adjust(left=0.10, bottom=0.10, right=0.98, top=0.96, wspace=0.05, hspace=0.05)

    for j, reg_data in enumerate(reg_datas):
        my_col = my_cols[j]
        pf = labels[j]
        lw0 = linewidths[j]
        if len(frame_inds)>0:
            try:
                image_ncc = np.array(reg_data['Image NCC'])[frame_inds]
            except:
                image_ncc = np.array(reg_data['NCC'])[frame_inds]
            frame_inds_loc = frame_inds.copy()
        else:
            try:
                image_ncc = np.array(reg_data['Image NCC'])
            except:
                image_ncc = np.array(reg_data['NCC'])
            frame_inds_loc = np.arange(len(image_ncc))

        axs3[1].plot(frame_inds_loc, image_ncc, c=my_col, linewidth=lw0)
        axs3[1].plot(image_ncc[0], c=my_col, linewidth=lw1, label=pf)
    axs3[1].grid(True)
    axs3[1].legend(fontsize=fs2)
    axs3[1].set_ylabel('Normalized Cross-Correlation', fontsize=fs)
    axs3[1].set_xlabel('Frame', fontsize=fs)
    axs3[1].set_ylim(ncc_min, ncc_max)
    axs3[1].set_xlim(fr_i, fr_a)
    axs3[0].axis(False)
    axs3[0].text(-0.1, 1.07, 'SIFT Registration Comparisons:  ' + data_dir, fontsize=fst3)
    llw1=0.3
    clw = [llw1, llw1]

    columns2 = ['NCC Mean', 'NCC Spread', 'SNR Mean', 'SNR Spread']
    cell2_text = []
    fig3_data = []
    limits = []
    rows = labels
    
    for j, (mean, spread) in enumerate(zip(means, spreads)):
        cell2_text.append(['{:.4f}'.format(mean[1]), '{:.4f}'.format(spread[1]),
                          '{:.4f}'.format(mean[2]), '{:.4f}'.format(spread[2])])
    n_cols = len(columns2)
    n_rows = len(rows)

    tbl2 = axs3[0].table(cellText = cell2_text,
                   rowLabels = rows,
                   colLabels = columns2,
                   cellLoc = 'center',
                   colLoc = 'center',
                   bbox = [0.45, 0, 0.55, 1.0],  # (left, bottom, width, height)
                  fontsize=16)
    rl = max([len(pf) for pf in labels])
    #tbl2.auto_set_column_width(col=[rl+5, len(columns2[0]), len(columns2[1]), len(columns2[2]), len(columns2[3])])
    tbl2.auto_set_column_width(col=list(range(len(columns2)+1)))
    tbl2.auto_set_font_size(False)

    table2_props = tbl2.properties()
    try:
        table2_cells = table2_props['child_artists']
    except:
        table2_cells = table2_props['children']

    tbl.auto_set_font_size(False)
    for j, cell in enumerate(table2_cells[0:n_cols*n_rows]):
        cell.get_text().set_color(my_cols[j//n_cols])
        cell.get_text().set_fontsize(fst3)
    for j, cell in enumerate(table2_cells[n_cols*(n_rows+1):]):
        cell.get_text().set_color(my_cols[j])
    for cell in table2_cells[n_cols*n_rows:]:
    #    cell.get_text().set_fontweight('bold')
        cell.get_text().set_fontsize(fst3)
    save_filename3 = save_filename.replace('.png', '_fig_and_table.png')
    if save_res_png:
        fig3.savefig(save_filename3, dpi=300)
    
    if save_res_png:
        # Generate a single multi-page CSV file
        xlsx_fname = save_filename.replace('.png', '.xlsx')
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(xlsx_fname, engine='xlsxwriter')
        fig2_df = pd.DataFrame(fig2_data, columns=columns)
        fig2_df.insert(0, '', labels)
        fig2_df.insert(1, 'Path', data_files)
        fig2_df.to_excel(writer, index=None, sheet_name='Summary')
        for reg_data, label in zip(tqdm(reg_datas, desc='saving the data into xlsx file'), labels):
            data_fn = label[0:31]
            reg_data.to_excel(writer, sheet_name=data_fn)
        #writer.save()
        writer.close()
    return xlsx_fname


#######################################
#    class FIBSEM_frame
#######################################

class FIBSEM_frame:
    """
    A class representing single FIB-SEM data frame.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    Contains the info/settings on a single FIB-SEM data frame and the procedures that can be performed on it.
    Initialization Parameters:
        Filename : string
            Filename of the file containing the FIBSEM frame

    Initialization kwargs:
        ftype : int
            0 - Shan Xu's binary format (default).  1 - tif files
        calculate_scaled_images : boolean
            Calculate Scaled Images from raw images using scalinfg data. Defauult is False
        use_dask_arrays : boolean
        memory_profiling : boolean
            If True will perfrom memory profiling. Default is False

    Attributes (only some more important are listed here)
    ----------
    fname : str
        filename of the individual data frame
    header : str
        1024 bytes - header
    FileVersion : int
        file version number
    ChanNum : int
        Number of channels
    EightBit : int
        8-bit data switch: 0 for 16-bit data, 1 for 8-bit data
    Scaling : 2D array of floats
        scaling parameters allowing to convert I16 data into actual electron counts 
    Sample_ID : str
        Sample_ID
    Notes : str
        Experiment notes
    DetA : str
        Detector A name
    DetB : str
        Detector B name ('None' if there is no Detector B)
    XResolution : int
        number of pixels - frame size in horizontal direction
    YResolution : int
        number of pixels - frame size in vertical direction
    PixelSize : float
        pixel size in nm. Default is 8.0

    Methods
    ----------
    print_header()
        Prints a formatted content of the file header.

    display_images()
        Display auto-scaled detector images without saving the figure into the file.

    save_images_jpeg(**kwargs)
        Display auto-scaled detector images and save the figure into JPEG file (s).

    save_images_tif(images_to_save = 'Both')
        Save the detector images into TIF file (s).

    get_image_min_max(image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False)
        Calculates the data range of the EM data.

    RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        Convert the Image A into 8-bit array.

    RawImageB_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
            Convert the Image B into 8-bit array.

    save_snapshot(**kwargs):
        Builds an image that contains both the Detector A and Detector B (if present) images as well as a table with important FIB-SEM parameters.

    analyze_noise_ROIs(Noise_ROIs, Hist_ROI ,**kwargs):
        Analyses the noise statistics in the selected ROI's of the EM data. (Calls Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):). This function is somewhat obsolete. When possible – use a method analyze_noise_statistics(**kwargs):, which calls Single_Image_Noise_Statistics(img, **kwargs):. That is a more powerful and accurate way of analysing the noise distribution.

    analyze_noise_statistics(**kwargs):
        Analyses the noise statistics of the EM data image. (Calls Single_Image_Noise_Statistics(img, **kwargs):).

    analyze_SNR_autocorr(**kwargs):
        Estimates SNR using auto-correlation analysis of a single image. (Calls Single_Image_SNR(img, **kwargs):).

    show_eval_box(**kwargs):
        Show the box used for evaluating the noise.

    determine_field_fattening_parameters(**kwargs):
        Perfrom 2D polynomial fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters.

    flatten_image(**kwargs):
        Flatten the image(s). Image flattening parameters must be pre-determined (determine_field_fattening_parameters).
    """

    def __init__(self, fname, **kwargs):
        '''
        Initialize FIBSEM_frame object.
        Parameters:
        ----------
        fname : string
            Filename (full path) of the file containing the FIBSEM frame

        kwargs:
        ----------
        ftype : int
            0 - Shan Xu's binary format (default).  1 - tif files
        memory_profiling : boolean
            Perform memory profiling during the data load and output it. Default is False.
        calculate_scaled_images : boolean
            Calculate Scaled Images from raw images using scaling data. Default is False.
        use_dask_arrays : boolean
            Load the data as DASK array

        '''
        memory_profiling = kwargs.get('memory_profiling', False)
        if memory_profiling:
            rss_before, vms_before, shared_before = get_process_memory()
            start_time = time.time()

        self.fname = fname
        def_ftype = 0
        fname_suff = Path(fname).suffix.lower()
        if fname_suff == '.tif' or fname_suff == '.tiff':
            def_ftype = 1
        #print('filename suffix: ',fname_suff, ', default filetype: ', def_ftype)
        self.ftype = kwargs.get("ftype", def_ftype) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        calculate_scaled_images_kw = kwargs.get('calculate_scaled_images', False)
        self.use_dask_arrays = kwargs.get("use_dask_arrays", False)
        if memory_profiling:
            elapsed_time = elapsed_since(start_time)
            rss_after, vms_after, shared_after = get_process_memory()
            print("Profiling: Start of Execution: RSS: {:>8} | VMS: {:>8} | SHR {"
                  ":>8} | time: {:>8}"
                .format(format_bytes(rss_after - rss_before),
                        format_bytes(vms_after - vms_before),
                        format_bytes(shared_after - shared_before),
                        elapsed_time))

    # for tif files
        if self.ftype == 1:
            self.RawImageA = tiff.imread(fname)
            self.FileVersion = -1
            self.DetA = 'Detector A'     # Name of detector A
            self.DetB = 'None'     # Name of detector B
            try:
                with tiff.TiffFile(fname) as tif:
                    tif_tags = {}
                    for tag in tif.pages[0].tags.values():
                        name, value = tag.name, tag.value
                        tif_tags[name] = value
                    self.header = tif_tags
                try:
                    if tif_tags['bits_per_sample'][0]==8:
                        self.EightBit = 1
                    else:
                        self.EightBit = 0
                except:
                    self.EightBit = int(type(self.RawImageA[0,0])==np.uint8)
            except:
                self.header = ''
                self.EightBit = int(type(self.RawImageA[0,0])==np.uint8)
            try:
                self.WD = tif_tags['helios_metadata']['EBeam']['WD']*1000.00 # Working Distance in mm
            except:
                pass
            try:
                self.EHT = tif_tags['helios_metadata']['EBeam']['HV']/1000.00 # EHT in kV
            except:
                pass
            try:
                self.SEMCurr = tif_tags['helios_metadata']['EBeam']['BeamCurrent'] # SEM probe current in A                  
            except:
                pass
            try:
                self.PixelSize = float(tif_tags['helios_metadata']['Scan']['PixelWidth']) * 1.0e9
            except:
                self.PixelSize = kwargs.get("PixelSize", 8.0)
            try:
                self.ScanRate = 1.0 / float(tif_tags['helios_metadata']['Scan']['Dwelltime'])
            except:
                pass
            try:
                self.MachineID = tif_tags['helios_metadata']['System']['SystemType'] + ' ' + str(tif_tags['helios_metadata']['System']['Dnumber'])
            except:
                pass
            self.Sample_ID = kwargs.get("Sample_ID", '')
            self.YResolution, self.XResolution = self.RawImageA.shape
            self.Scaling = np.array([[1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]]).T
            if memory_profiling:
                elapsed_time = elapsed_since(start_time)
                rss_after, vms_after, shared_after = get_process_memory()
                print("Profiling: Finished TIFF Read: RSS: {:>8} | VMS: {:>8} | SHR {"
                      ":>8} | time: {:>8}"
                    .format(format_bytes(rss_after - rss_before),
                            format_bytes(vms_after - vms_before),
                            format_bytes(shared_after - shared_before),
                            elapsed_time))

    # for Shan Xu's data files 
        if self.ftype == 0:
            self.SaveOversamples = 0
            fid = open(self.fname, "rb")
            fid.seek(0, 0)
            self.header = fid.read(1024) # Read in self.header
            self.FileMagicNum = unpack('>L',self.header[0:4])[0]                       # Read in magic number, should be 3555587570
            self.FileVersion = unpack('>h',self.header[4:6])[0]                        # Read in file version number
            self.FileType = unpack('>h',self.header[6:8])[0]                           # Read in file type, 1 is Zeiss Neon detectors
            self.SWdate = (unpack('10s',self.header[8:18])[0]).decode("utf-8")         # Read in SW date
            self.TimeStep = unpack('>d',self.header[24:32])[0]                         # Read in AI sampling time (including oversampling) in seconds
            self.ChanNum = unpack('b',self.header[32:33])[0]                           # Read in number of channels
            self.EightBit = unpack('b',self.header[33:34])[0]                          # Read in 8-bit data switch

            if self.FileVersion == 1:
                # Read in self.AI channel self.Scaling factors, (col#: self.AI#), (row#: offset, gain, 2nd order, 3rd order)
                self.ScalingS = unpack('>'+str(4*self.ChanNum)+'d',self.header[36:(36+self.ChanNum*32)])
            elif self.FileVersion == 2 or self.FileVersion == 3 or self.FileVersion == 4 or self.FileVersion == 5 or self.FileVersion == 6:
                self.ScalingS = unpack('>'+str(4*self.ChanNum)+'f',self.header[36:(36+self.ChanNum*16)])
            else:
                self.ScalingS = unpack('>8f',self.header[36:68])
                self.Scaling = np.transpose(np.asarray(self.ScalingS).reshape(2,4))
            
            if self.FileVersion > 8 :
                self.RestartFlag = unpack('b',self.header[68:69])[0]              # Read in restart flag
                self.StageMove = unpack('b',self.header[69:70])[0]                # Read in stage move flag
                self.FirstPixelX = unpack('>l',self.header[70:74])[0]              # Read in first pixel X coordinate (center = 0)
                self.FirstPixelY = unpack('>l',self.header[74:78])[0]              # Read in first pixel Y coordinate (center = 0)
            
            self.XResolution = unpack('>L',self.header[100:104])[0]                # Read X resolution
            self.YResolution = unpack('>L',self.header[104:108])[0]                # Read Y resolution

            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                self.Oversampling = unpack('>B',self.header[108:109])[0]               # self.AI oversampling     
                self.AIDelay = unpack('>h',self.header[109:111])[0]                    # self.AI delay (# of samples)
            else:
                self.Oversampling = unpack('>H',self.header[108:110])[0]

            self.ZeissScanSpeed = unpack('b',self.header[111:112])[0] # Scan speed (Zeiss #)

            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                self.ScanRate = unpack('>d',self.header[112:120])[0]                   # Actual AO (scanning) rate    
                self.FramelineRampdownRatio = unpack('>d',self.header[120:128])[0]     # Frameline rampdown ratio
                self.Xmin = unpack('>d',self.header[128:136])[0]                       # X coil minimum voltage
                self.Xmax = unpack('>d',self.header[136:144])[0]                       # X coil maximum voltage
                self.Detmin = -10                                                      # Detector minimum voltage
                self.Detmax = 10                                                       # Detector maximum voltage
            else:
                self.ScanRate = unpack('>f',self.header[112:116])[0]                   # Actual AO (scanning) rate    
                self.FramelineRampdownRatio = unpack('>f',self.header[116:120])[0]     # Frameline rampdown ratio
                self.Xmin = unpack('>f',self.header[120:124])[0]                       # X coil minimum voltage
                self.Xmax = unpack('>f',self.header[124:128])[0]                       # X coil maximum voltage
                self.Detmin = unpack('>f',self.header[128:132])[0]                     # Detector minimum voltage
                self.Detmax = unpack('>f',self.header[132:136])[0]                     # Detector maximum voltage
                self.DecimatingFactor = unpack('>H',self.header[136:138])[0]           # Decimating factor

            self.SaveOversamples = unpack('b',self.header[138:139])[0]                  # Average Oversamples (normal, False) or Save All Oversamples (for noise stdies, True)
            self.AI1 = unpack('b',self.header[151:152])[0]                              # self.AI Ch1
            self.AI2 = unpack('b',self.header[152:153])[0]                              # self.AI Ch2 
            self.AI3 = unpack('b',self.header[153:154])[0]                              # self.AI Ch3
            self.AI4 = unpack('b',self.header[154:155])[0]                              # self.AI Ch4
            
            self.Notes = (unpack('200s',self.header[180:380])[0]).decode("utf-8")       # Read in notes

            if self.FileVersion > 8 :
                self.Sample_ID = (unpack('25s',self.header[155:180])[0]).decode("utf-8").strip('\x00') # Read in Sample ID
            else:
                self.Sample_ID = self.Notes.split(',')[0].strip('\x00')

            if self.FileVersion == 1 or self.FileVersion == 2:
                self.DetA = (unpack('10s',self.header[380:390])[0]).decode("utf-8")     # Name of detector A
                self.DetB = (unpack('18s',self.header[390:408])[0]).decode("utf-8")     # Name of detector B
                self.DetC = (unpack('20s',self.header[700:720])[0]).decode("utf-8")     # Name of detector C
                self.DetD = (unpack('20s',self.header[720:740])[0]).decode("utf-8")     # Name of detector D
                self.Mag = unpack('>d',self.header[408:416])[0]                         # Magnification
                self.PixelSize = unpack('>d',self.header[416:424])[0]                   # Pixel size in nm
                self.WD = unpack('>d',self.header[424:432])[0]                          # Working Distance in mm
                self.EHT = unpack('>d',self.header[432:440])[0]                         # EHT in kV
                self.SEMApr = unpack('b',self.header[440:441])[0]                       # SEM aperture number
                self.HighCurrent = unpack('b',self.header[441:442])[0]                  # high current mode (1=on, 0=off)
                self.SEMCurr = unpack('>d',self.header[448:456])[0]                     # SEM probe current in A
                self.SEMRot = unpack('>d',self.header[456:464])[0]                      # SEM scan roation in degree
                self.ChamVac = unpack('>d',self.header[464:472])[0]                     # Chamber vacuum
                self.GunVac = unpack('>d',self.header[472:480])[0]                      # E-gun vacuum
                self.SEMStiX = unpack('>d',self.header[480:488])[0]                     # SEM stigmation X
                self.SEMStiY = unpack('>d',self.header[488:496])[0]                     # SEM stigmation Y
                self.SEMAlnX = unpack('>d',self.header[496:504])[0]                     # SEM aperture alignment X
                self.SEMAlnY = unpack('>d',self.header[504:512])[0]                     # SEM aperture alignment Y
                self.StageX = unpack('>d',self.header[512:520])[0]                      # Stage position X in mm
                self.StageY = unpack('>d',self.header[520:528])[0]                      # Stage position Y in mm
                self.StageZ = unpack('>d',self.header[528:536])[0]                      # Stage position Z in mm
                self.StageT = unpack('>d',self.header[536:544])[0]                      # Stage position T in degree
                self.StageR = unpack('>d',self.header[544:552])[0]                      # Stage position R in degree
                self.StageM = unpack('>d',self.header[552:560])[0]                      # Stage position M in mm
                self.BrightnessA = unpack('>d',self.header[560:568])[0]                 # Detector A brightness (%)
                self.ContrastA = unpack('>d',self.header[568:576])[0]                   # Detector A contrast (%)
                self.BrightnessB = unpack('>d',self.header[576:584])[0]                 # Detector B brightness (%)
                self.ContrastB = unpack('>d',self.header[584:592])[0]                   # Detector B contrast (%)
                self.Mode = unpack('b',self.header[600:601])[0]                         # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                self.FIBFocus = unpack('>d',self.header[608:616])[0]                    # FIB focus in kV
                self.FIBProb = unpack('b',self.header[616:617])[0]                      # FIB probe number
                self.FIBCurr = unpack('>d',self.header[624:632])[0]                     # FIB emission current
                self.FIBRot = unpack('>d',self.header[632:640])[0]                      # FIB scan rotation
                self.FIBAlnX = unpack('>d',self.header[640:648])[0]                     # FIB aperture alignment X
                self.FIBAlnY = unpack('>d',self.header[648:656])[0]                     # FIB aperture alignment Y
                self.FIBStiX = unpack('>d',self.header[656:664])[0]                     # FIB stigmation X
                self.FIBStiY = unpack('>d',self.header[664:672])[0]                     # FIB stigmation Y
                self.FIBShiftX = unpack('>d',self.header[672:680])[0]                   # FIB beam shift X in micron
                self.FIBShiftY = unpack('>d',self.header[680:688])[0]                   # FIB beam shift Y in micron
            else:
                self.DetA = (unpack('10s',self.header[380:390])[0]).decode("utf-8")     # Name of detector A
                self.DetB = (unpack('18s',self.header[390:408])[0]).decode("utf-8")     # Name of detector B
                self.DetC = (unpack('20s',self.header[410:430])[0]).decode("utf-8")     # Name of detector C
                self.DetD = (unpack('20s',self.header[430:450])[0]).decode("utf-8")     # Name of detector D
                self.Mag = unpack('>f',self.header[460:464])[0]                         # Magnification
                self.PixelSize = unpack('>f',self.header[464:468])[0]                   # Pixel size in nm
                self.WD = unpack('>f',self.header[468:472])[0]                          # Working Distance in mm
                self.EHT = unpack('>f',self.header[472:476])[0]                         # EHT in kV
                self.SEMApr = unpack('b',self.header[480:481])[0]                       # SEM aperture number
                self.HighCurrent = unpack('b',self.header[481:482])[0]                  # high current mode (1=on, 0=off)
                self.SEMCurr = unpack('>f',self.header[490:494])[0]                     # SEM probe current in A
                self.SEMRot = unpack('>f',self.header[494:498])[0]                      # SEM scan roation in degree
                self.ChamVac = unpack('>f',self.header[498:502])[0]                     # Chamber vacuum
                self.GunVac = unpack('>f',self.header[502:506])[0]                      # E-gun vacuum
                self.SEMShiftX = unpack('>f',self.header[510:514])[0]                   # SEM beam shift X
                self.SEMShiftY = unpack('>f',self.header[514:518])[0]                   # SEM beam shift Y
                self.SEMStiX = unpack('>f',self.header[518:522])[0]                     # SEM stigmation X
                self.SEMStiY = unpack('>f',self.header[522:526])[0]                     # SEM stigmation Y
                self.SEMAlnX = unpack('>f',self.header[526:530])[0]                     # SEM aperture alignment X
                self.SEMAlnY = unpack('>f',self.header[530:534])[0]                     # SEM aperture alignment Y
                self.StageX = unpack('>f',self.header[534:538])[0]                      # Stage position X in mm
                self.StageY = unpack('>f',self.header[538:542])[0]                      # Stage position Y in mm
                self.StageZ = unpack('>f',self.header[542:546])[0]                      # Stage position Z in mm
                self.StageT = unpack('>f',self.header[546:550])[0]                      # Stage position T in degree
                self.StageR = unpack('>f',self.header[550:554])[0]                      # Stage position R in degree
                self.StageM = unpack('>f',self.header[554:558])[0]                      # Stage position M in mm
                self.BrightnessA = unpack('>f',self.header[560:564])[0]                 # Detector A brightness (%)
                self.ContrastA = unpack('>f',self.header[564:568])[0]                   # Detector A contrast (%)
                self.BrightnessB = unpack('>f',self.header[568:572])[0]                 # Detector B brightness (%)
                self.ContrastB = unpack('>f',self.header[572:576])[0]                   # Detector B contrast (%)
                self.Mode = unpack('b',self.header[600:601])[0]                         # FIB mode: 0=SEM, 1=FIB, 2=Milling, 3=SEM+FIB, 4=Mill+SEM, 5=SEM Drift Correction, 6=FIB Drift Correction, 7=No Beam, 8=External, 9=External+SEM
                self.FIBFocus = unpack('>f',self.header[604:608])[0]                    # FIB focus in kV
                self.FIBProb = unpack('b',self.header[608:609])[0]                      # FIB probe number
                self.FIBCurr = unpack('>f',self.header[620:624])[0]                     # FIB emission current
                self.FIBRot = unpack('>f',self.header[624:628])[0]                      # FIB scan rotation
                self.FIBAlnX = unpack('>f',self.header[628:632])[0]                     # FIB aperture alignment X
                self.FIBAlnY = unpack('>f',self.header[632:636])[0]                     # FIB aperture alignment Y
                self.FIBStiX = unpack('>f',self.header[636:640])[0]                     # FIB stigmation X
                self.FIBStiY = unpack('>f',self.header[640:644])[0]                     # FIB stigmation Y
                self.FIBShiftX = unpack('>f',self.header[644:648])[0]                   # FIB beam shift X in micron
                self.FIBShiftY = unpack('>f',self.header[648:652])[0]                   # FIB beam shift Y in micron

            if self.FileVersion > 4:
                self.MillingXResolution = unpack('>L',self.header[652:656])[0]                       # FIB milling X resolution
                self.MillingYResolution = unpack('>L',self.header[656:660])[0]                       # FIB milling Y resolution
                self.MillingXSize = unpack('>f',self.header[660:664])[0]                             # FIB milling X size (um)
                self.MillingYSize = unpack('>f',self.header[664:668])[0]                             # FIB milling Y size (um)
                self.MillingULAng = unpack('>f',self.header[668:672])[0]                             # FIB milling upper left inner angle (deg)
                self.MillingURAng = unpack('>f',self.header[672:676])[0]                             # FIB milling upper right inner angle (deg)
                self.MillingLineTime = unpack('>f',self.header[676:680])[0]                          # FIB line milling time (s)
                self.FIBFOV = unpack('>f',self.header[680:684])[0]                                   # FIB FOV (um)
                self.MillingLinesPerImage = unpack('>H',self.header[684:686])[0]                     # FIB milling lines per image
                self.MillingPIDOn = unpack('>b',self.header[686:687])[0]                             # FIB milling PID on
                self.MillingPIDMeasured = unpack('>b',self.header[689:690])[0]                       # FIB milling PID measured (0:specimen, 1:beamdump)
                self.MillingPIDTarget = unpack('>f',self.header[690:694])[0]                         # FIB milling PID target
                self.MillingPIDTargetSlope = unpack('>f',self.header[694:698])[0]                    # FIB milling PID target slope
                self.MillingPIDP = unpack('>f',self.header[698:702])[0]                              # FIB milling PID P
                self.MillingPIDI = unpack('>f',self.header[702:706])[0]                              # FIB milling PID I
                self.MillingPIDD = unpack('>f',self.header[706:710])[0]                              # FIB milling PID D
                self.MachineID = (unpack('30s',self.header[800:830])[0]).decode("utf-8")             # Machine ID
                self.SEMSpecimenI = unpack('>f',self.header[672:676])[0]                             # SEM specimen current (nA)

            if self.FileVersion > 5 :
                self.Temperature = unpack('>f',self.header[850:854])[0]                              # Temperature (F)
                self.FaradayCupI = unpack('>f',self.header[854:858])[0]                              # Faraday cup current (nA)
                self.FIBSpecimenI = unpack('>f',self.header[858:862])[0]                             # FIB specimen current (nA)
                self.BeamDump1I = unpack('>f',self.header[862:866])[0]                               # Beam dump 1 current (nA)
                self.SEMSpecimenI = unpack('>f',self.header[866:870])[0]                             # SEM specimen current (nA)
                self.MillingYVoltage = unpack('>f',self.header[870:874])[0]                          # Milling Y voltage (V)
                self.FocusIndex = unpack('>f',self.header[874:878])[0]                               # Focus index
                self.FIBSliceNum = unpack('>L',self.header[878:882])[0]                              # FIB slice #

            if self.FileVersion > 7:
                self.BeamDump2I = unpack('>f',self.header[882:886])[0]                              # Beam dump 2 current (nA)
                self.MillingI = unpack('>f',self.header[886:890])[0]                                # Milling current (nA)

            self.FileLength = unpack('>q',self.header[1000:1008])[0]                                # Read in file length in bytes 

            if memory_profiling:
                elapsed_time = elapsed_since(start_time)
                rss_after, vms_after, shared_after = get_process_memory()
                print("Profiling: Finish Header Read: RSS: {:>8} | VMS: {:>8} | SHR {"
                      ":>8} | time: {:>8}"
                    .format(format_bytes(rss_after - rss_before),
                            format_bytes(vms_after - vms_before),
                            format_bytes(shared_after - shared_before),
                            elapsed_time))            
#                Finish self.header read
#
#                Read raw data
#                fid.seek(1024, 0)
#                n_elements = self.ChanNum * self.XResolution * self.YResolution
#                print(n_elements, self.ChanNum, self.XResolution, self.YResolution)
#                if self.EightBit==1:
#                    raw_data = fid.read(n_elements) # Read in data
#                    Raw = unpack('>'+str(n_elements)+'B',raw_data)
#                else:
#                    #raw_data = fid.read(2*n_elements) # Read in data
#                    #Raw = unpack('>'+str(n_elements)+'h',raw_data)
#                fid.close()
#                finish reading raw data

            n_elements = self.ChanNum * self.XResolution * self.YResolution
            if self.SaveOversamples:
                n_elements *= self.Oversampling
            fid.seek(1024, 0)
            if self.EightBit==1:
                dt = np.dtype(np.uint8)
                dt = dt.newbyteorder('>')
                if self.use_dask_arrays:
                    Raw = da.from_array(np.frombuffer(fid.read(n_elements), dtype=dt))
                else:
                    Raw = np.frombuffer(fid.read(n_elements), dtype=dt)
            else:
                dt = np.dtype(np.int16)
                dt = dt.newbyteorder('>')
                if self.use_dask_arrays:
                    Raw = da.from_array(np.frombuffer(fid.read(2*n_elements),dtype=dt))
                else:
                    Raw = np.frombuffer(fid.read(2*n_elements),dtype=dt)
            fid.close()
            if memory_profiling:
                elapsed_time = elapsed_since(start_time)
                rss_after, vms_after, shared_after = get_process_memory()
                print("Profiling: Finished File Read: RSS: {:>8} | VMS: {:>8} | SHR {"
                      ":>8} | time: {:>8}"
                    .format(format_bytes(rss_after - rss_before),
                            format_bytes(vms_after - vms_before),
                            format_bytes(shared_after - shared_before),
                            elapsed_time))   
            # finish reading raw data
     
            if self.SaveOversamples:
                Raw = np.moveaxis(np.array(Raw).reshape(self.YResolution, self.XResolution, self.Oversampling, self.ChanNum), 2, 3)
            else:
                try:
                    Raw = np.array(Raw).reshape(self.YResolution, self.XResolution, self.ChanNum)
                except:
                    # handling the case of incomplete data write/read
                    Raw_imcomplete = np.array(Raw)
                    Raw_complete = np.zeros(self.YResolution*self.XResolution*self.ChanNum, dtype=dt)
                    Raw_complete[:len(Raw_imcomplete)] = Raw_imcomplete
                    Raw_complete[len(Raw_imcomplete):] = Raw_imcomplete[-1]
                    Raw = np.array(Raw_complete).reshape(self.YResolution, self.XResolution, self.ChanNum)

            #print(np.shape(Raw), type(Raw), type(Raw[0,0]))

            #data = np.asarray(datab).reshape(self.YResolution,self.XResolution,ChanNum)
            if self.EightBit == 1:
                if self.AI1 == 1:
                    self.RawImageA = Raw[:,:,0]
                    #self.ImageA = (Raw[:,:,0].astype(np.float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(np.int32)
                    if self.AI2 == 1:
                        self.RawImageB = Raw[:,:,1]
                        #self.ImageB = (Raw[:,:,1].astype(np.float32)*self.ScanRate/self.Scaling[0,1]/self.Scaling[2,1]/self.Scaling[3,1]+self.Scaling[1,1]).astype(np.int32)
                elif self.AI2 == 1:
                    self.RawImageB = Raw[:,:,0]
                    #self.ImageB = (Raw[:,:,0].astype(np.float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(np.int32)
            else:
                if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3 or self.FileVersion == 4 or self.FileVersion == 5 or self.FileVersion == 6:
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:,:,0]
                        #self.ImageA = self.Scaling[0,0] + self.RawImageA * self.Scaling[1,0]  # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:,:,1]
                            #self.ImageB = self.Scaling[0,1] + self.RawImageB * self.Scaling[1,1]
                            if self.AI3 == 1:
                                self.RawImageC = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                #self.ImageC = self.Scaling[0,2] + self.RawImageC * self.Scaling[1,2]
                                if self.AI4 == 1:
                                    self.RawImageD = (Raw[:,:,3]).reshape(self.YResolution,self.XResolution)
                                    #self.ImageD = self.Scaling[0,3] + self.RawImageD * self.Scaling[1,3]
                            elif self.AI4 == 1:
                                self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                #self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI3 == 1:
                            self.RawImageC = Raw[:,:,1]
                            #self.ImageC = self.Scaling[0,1] + self.RawImageC * self.Scaling[1,1]
                            if self.AI4 == 1:
                                self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                #self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI4 == 1:
                            self.RawImageD = Raw[:,:,1]
                            #self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:,:,0]
                        #self.ImageB = self.Scaling[0,0] + self.RawImageB * self.Scaling[1,0]
                        if self.AI3 == 1:
                            self.RawImageC = Raw[:,:,1]
                            #self.ImageC = self.Scaling[0,1] + self.RawImageC * self.Scaling[1,1]
                            if self.AI4 == 1:
                                self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                #self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI4 == 1:
                            self.RawImageD = Raw[:,:,1]
                            #self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI3 == 1:
                        self.RawImageC = Raw[:,:,0]
                        #self.ImageC = self.Scaling[0,0] + self.RawImageC * self.Scaling[1,0]
                        if self.AI4 == 1:
                            self.RawImageD = Raw[:,:,1]
                            #self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI4 == 1:
                        self.RawImageD = Raw[:,:,0]
                        #self.ImageD = self.Scaling[0,0] + self.RawImageD * self.Scaling[1,0]
                        
                elif self.FileVersion == 7:
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:,:,0]
                        #self.ImageA = (self.RawImageA - self.Scaling[1,0])*self.Scaling[2,0]
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:,:,1]
                            #self.ImageB = (self.RawImageB - self.Scaling[1,1])*self.Scaling[2,1]
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:,:,0]
                        #self.ImageB = (self.RawImageB - self.Scaling[1,1])*self.Scaling[2,1]
                        
                elif  self.FileVersion == 8 or self.FileVersion == 9:
                    self.ElectronFactor1 = 0.1;             # 16-bit intensity is 10x electron counts
                    self.Scaling[3,0] = self.ElectronFactor1
                    self.ElectronFactor2 = 0.1;             # 16-bit intensity is 10x electron counts
                    self.Scaling[3,1] = self.ElectronFactor2
                    if self.AI1 == 1:
                        self.RawImageA = Raw[:,:,0]
                        #self.ImageA = (self.RawImageA - self.Scaling[1,0]) * self.Scaling[2,0] / self.ScanRate * self.Scaling[0,0] / self.ElectronFactor1                        
                        # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            self.RawImageB = Raw[:,:,1]
                            #self.ImageB = (self.RawImageB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2
                    elif self.AI2 == 1:
                        self.RawImageB = Raw[:,:,0]
                        #self.ImageB = (self.RawImageB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2

            if self.SaveOversamples:
                self.RawSamplesA = self.RawImageA.copy()
                self.RawSamplesB = self.RawImageB.copy()
                #self.SamplesA = self.ImageA.copy()
                #self.SamplesB = self.ImageB.copy()
                self.RawImageA = np.mean(self.RawSamplesA, axis=2)
                self.RawImageB = np.mean(self.RawSamplesB, axis=2)
                #self.ImageA = np.mean(self.SamplesA, axis=2)
                #self.ImageB = np.mean(self.SamplesB, axis=2)
            del Raw
            if memory_profiling:
                elapsed_time = elapsed_since(start_time)
                rss_after, vms_after, shared_after = get_process_memory()
                print("Profiling: Finished Raw Conv.: RSS: {:>8} | VMS: {:>8} | SHR {"
                      ":>8} | time: {:>8}"
                    .format(format_bytes(rss_after - rss_before),
                            format_bytes(vms_after - vms_before),
                            format_bytes(shared_after - shared_before),
                            elapsed_time))   

            if calculate_scaled_images_kw:
                self.calculate_scaled_images()
                if memory_profiling:
                    elapsed_time = elapsed_since(start_time)
                    rss_after, vms_after, shared_after = get_process_memory()
                    print("Profiling: Calc.Scaled Images: RSS: {:>8} | VMS: {:>8} | SHR {"
                          ":>8} | time: {:>8}"
                        .format(format_bytes(rss_after - rss_before),
                                format_bytes(vms_after - vms_before),
                                format_bytes(shared_after - shared_before),
                                elapsed_time))  
            if memory_profiling:
                elapsed_time = elapsed_since(start_time)
                rss_after, vms_after, shared_after = get_process_memory()
                print("Profiling: Finished Execution: RSS: {:>8} | VMS: {:>8} | SHR {"
                      ":>8} | time: {:>8}"
                    .format(format_bytes(rss_after - rss_before),
                            format_bytes(vms_after - vms_before),
                            format_bytes(shared_after - shared_before),
                            elapsed_time))   

    def calculate_scaled_images(self):
        '''
        Calculate Scaled Imaged from RawImages using Scaling Data.
        '''
        if self.ftype == 0:
            if self.EightBit == 1:
                if self.AI1 == 1:
                    #self.RawImageA = Raw[:,:,0]
                    self.ImageA = (self.RawImageA.astype(np.float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(np.int32)
                    if self.AI2 == 1:
                        #self.RawImageB = Raw[:,:,1]
                        self.ImageB = (self.RawImageB.astype(np.float32)*self.ScanRate/self.Scaling[0,1]/self.Scaling[2,1]/self.Scaling[3,1]+self.Scaling[1,1]).astype(np.int32)
                elif self.AI2 == 1:
                    #self.RawImageB = Raw[:,:,0]
                    self.ImageB = (self.RawImageB.astype(np.float32)*self.ScanRate/self.Scaling[0,0]/self.Scaling[2,0]/self.Scaling[3,0]+self.Scaling[1,0]).astype(np.int32)
            else:
                if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3 or self.FileVersion == 4 or self.FileVersion == 5 or self.FileVersion == 6:
                    if self.AI1 == 1:
                        #self.RawImageA = Raw[:,:,0]
                        self.ImageA = self.Scaling[0,0] + self.RawImageA * self.Scaling[1,0]  # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            #self.RawImageB = Raw[:,:,1]
                            self.ImageB = self.Scaling[0,1] + self.RawImageB * self.Scaling[1,1]
                            if self.AI3 == 1:
                                #self.RawImageC = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageC = self.Scaling[0,2] + self.RawImageC * self.Scaling[1,2]
                                if self.AI4 == 1:
                                    #self.RawImageD = (Raw[:,:,3]).reshape(self.YResolution,self.XResolution)
                                    self.ImageD = self.Scaling[0,3] + self.RawImageD * self.Scaling[1,3]
                            elif self.AI4 == 1:
                                #self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI3 == 1:
                            #self.RawImageC = Raw[:,:,1]
                            self.ImageC = self.Scaling[0,1] + self.RawImageC * self.Scaling[1,1]
                            if self.AI4 == 1:
                                #self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI4 == 1:
                            #self.RawImageD = Raw[:,:,1]
                            self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI2 == 1:
                        #self.RawImageB = Raw[:,:,0]
                        self.ImageB = self.Scaling[0,0] + self.RawImageB * self.Scaling[1,0]
                        if self.AI3 == 1:
                            #self.RawImageC = Raw[:,:,1]
                            self.ImageC = self.Scaling[0,1] + self.RawImageC * self.Scaling[1,1]
                            if self.AI4 == 1:
                                #self.RawImageD = (Raw[:,:,2]).reshape(self.YResolution,self.XResolution)
                                self.ImageD = self.Scaling[0,2] + self.RawImageD * self.Scaling[1,2]
                        elif self.AI4 == 1:
                            #self.RawImageD = Raw[:,:,1]
                            self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI3 == 1:
                        #self.RawImageC = Raw[:,:,0]
                        self.ImageC = self.Scaling[0,0] + self.RawImageC * self.Scaling[1,0]
                        if self.AI4 == 1:
                            #self.RawImageD = Raw[:,:,1]
                            self.ImageD = self.Scaling[0,1] + self.RawImageD * self.Scaling[1,1]
                    elif self.AI4 == 1:
                        #self.RawImageD = Raw[:,:,0]
                        self.ImageD = self.Scaling[0,0] + self.RawImageD * self.Scaling[1,0]
                        
                elif self.FileVersion == 7:
                    if self.AI1 == 1:
                        #self.RawImageA = Raw[:,:,0]
                        self.ImageA = (self.RawImageA - self.Scaling[1,0])*self.Scaling[2,0]
                        if self.AI2 == 1:
                            #self.RawImageB = Raw[:,:,1]
                            self.ImageB = (self.RawImageB - self.Scaling[1,1])*self.Scaling[2,1]
                    elif self.AI2 == 1:
                        #self.RawImageB = Raw[:,:,0]
                        self.ImageB = (self.RawImageB - self.Scaling[1,1])*self.Scaling[2,1]
                        
                elif  self.FileVersion == 8 or self.FileVersion == 9:
                    self.ElectronFactor1 = 0.1;             # 16-bit intensity is 10x electron counts
                    self.Scaling[3,0] = self.ElectronFactor1
                    self.ElectronFactor2 = 0.1;             # 16-bit intensity is 10x electron counts
                    self.Scaling[3,1] = self.ElectronFactor2
                    if self.AI1 == 1:
                        #self.RawImageA = Raw[:,:,0]
                        self.ImageA = (self.RawImageA - self.Scaling[1,0]) * self.Scaling[2,0] / self.ScanRate * self.Scaling[0,0] / self.ElectronFactor1
                        if self.SaveOversamples:
                            self.SamplesA = (self.RawSamplesA - self.Scaling[1,0]) * self.Scaling[2,0] / self.ScanRate * self.Scaling[0,0] / self.ElectronFactor1
                        # Converts raw I16 data to voltage based on self.Scaling factors
                        if self.AI2 == 1:
                            #self.RawImageB = Raw[:,:,1]
                            self.ImageB = (self.RawImageB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2
                            if self.SaveOversamples:
                                self.SamplesB = (self.RawSamplesB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2
                        # Converts raw I16 data to voltage based on self.Scaling factors
                    elif self.AI2 == 1:
                        #self.RawImageB = Raw[:,:,0]
                        self.ImageB = (self.RawImageB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2
                        if self.SaveOversamples:
                            self.SamplesB = (self.RawSamplesB - self.Scaling[1,1]) * self.Scaling[2,1] / self.ScanRate * self.Scaling[0,1] / self.ElectronFactor2


    def print_header(self):
        '''
        Prints a formatted content of the file header.

        '''
        if self.FileVersion == -1 :
            print('Sample_ID=', self.Sample_ID)
            print('DetA=', self.DetA)
            print('DetB=', self.DetB)
            print('EightBit=', self.EightBit)
            print('XResolution=', self.XResolution)
            print('YResolution=', self.YResolution)
            print('PixelSize=', self.PixelSize)
        else:
            print('FileMagicNum=', self.FileMagicNum)
            print('FileVersion=', self.FileVersion)
            print('FileType=', self.FileType)
            print('SWdate=', self.SWdate)
            print('TimeStep=', self.TimeStep)                
            print('ChanNum=', self.ChanNum)
            print('EightBit=', self.EightBit)
            print('Scaling=', self.Scaling)
            if self.FileVersion > 8 :
                print('RestartFlag=', self.RestartFlag)
                print('StageMove=', self.StageMove)
                print('FirstPixelX=', self.FirstPixelX)
                print('FirstPixelY=', self.FirstPixelY)
            print('XResolution=', self.XResolution)
            print('YResolution=', self.YResolution)
            if self.FileVersion == 1 or self.FileVersion == 2 or self.FileVersion == 3:
                print('AIDelay=', self.AIDelay)
            print('Oversampling=', self.Oversampling)
            print('ZeissScanSpeed=', self.ZeissScanSpeed)
            print('DecimatingFactor=', self.DecimatingFactor)
            print('SaveOversamples=', self.SaveOversamples )
            print('ScanRate=', self.ScanRate)
            print('FramelineRampdownRatio=', self.FramelineRampdownRatio)
            print('Xmin=', self.Xmin)
            print('Xmax=', self.Xmax)
            print('Detmin=', self.Detmin)
            print('Detmax=', self.Detmax)
            print('AI1=', self.AI1)
            print('AI2=', self.AI2)
            print('AI3=', self.AI3)
            print('AI4=', self.AI4)
            if self.FileVersion > 8 :
                 print('Sample_ID=', self.Sample_ID)
            print('Notes=', self.Notes)
            print('SEMShiftX=', self.SEMShiftX)
            print('SEMShiftY=', self.SEMShiftY)
            print('DetA=', self.DetA)
            print('DetB=', self.DetB)
            print('DetC=', self.DetC)
            print('DetD=', self.DetD)
            print('Mag=', self.Mag)
            print('PixelSize=', self.PixelSize)
            print('WD=', self.WD)
            print('EHT=', self.EHT)
            print('SEMApr=', self.SEMApr)
            print('HighCurrent=', self.HighCurrent)
            print('SEMCurr=', self.SEMCurr)
            print('SEMRot=', self.SEMRot)
            print('ChamVac=', self.ChamVac)
            print('GunVac=', self.GunVac)
            print('SEMStiX=', self.SEMStiX)
            print('SEMStiY=', self.SEMStiY)
            print('SEMAlnX=', self.SEMAlnX)
            print('SEMAlnY=', self.SEMAlnY)
            print('StageX=', self.StageX)
            print('StageY=', self.StageY)
            print('StageZ=', self.StageZ)
            print('StageT=', self.StageT)
            print('StageR=', self.StageR)
            print('StageM=', self.StageM)
            print('BrightnessA=', self.BrightnessA)
            print('ContrastA=', self.ContrastA)
            print('BrightnessB=', self.BrightnessB)
            print('ContrastB=', self.ContrastB)
            print('Mode=', self.Mode)
            print('FIBFocus=', self.FIBFocus)
            print('FIBProb=', self.FIBProb)
            print('FIBCurr=', self.FIBCurr)
            print('FIBRot=', self.FIBRot)
            print('FIBAlnX=', self.FIBAlnX)
            print('FIBAlnY=', self.FIBAlnY)
            print('FIBStiX=', self.FIBStiX)
            print('FIBStiY=', self.FIBStiY)
            print('FIBShiftX=', self.FIBShiftX)
            print('FIBShiftY=', self.FIBShiftY)
            if self.FileVersion > 4:
                print('MillingXResolution=', self.MillingXResolution)
                print('MillingYResolution=', self.MillingYResolution)
                print('MillingXSize=', self.MillingXSize)
                print('MillingYSize=', self.MillingYSize)
                print('MillingULAng=', self.MillingULAng)
                print('MillingURAng=', self.MillingURAng)
                print('MillingLineTime=', self.MillingLineTime)
                print('FIBFOV (um)=', self.FIBFOV)
                print('MillingPIDOn=', self.MillingPIDOn)
                print('MillingPIDMeasured=', self.MillingPIDMeasured)
                print('MillingPIDTarget=', self.MillingPIDTarget)
                print('MillingPIDTargetSlope=', self.MillingPIDTargetSlope)
                print('MillingPIDP=', self.MillingPIDP)
                print('MillingPIDI=', self.MillingPIDI)
                print('MillingPIDD=', self.MillingPIDD)
                print('MachineID=', self.MachineID)
                print('SEMSpecimenI=', self.SEMSpecimenI)
            if self.FileVersion > 5:
                print('Temperature=', self.Temperature)
                print('FaradayCupI=', self.FaradayCupI)
                print('FIBSpecimenI=', self.FIBSpecimenI)
                print('BeamDump1I=', self.BeamDump1I)
                print('MillingYVoltage=', self.MillingYVoltage)
                print('FocusIndex=', self.FocusIndex)
                print('FIBSliceNum=', self.FIBSliceNum)
            if self.FileVersion > 7:
                print('BeamDump2I=', self.BeamDump2I)
                print('MillingI=', self.MillingI)
            print('SEMSpecimenI=', self.SEMSpecimenI)   
            print('FileLength=', self.FileLength)
    
    def display_images(self):
        '''
        Display auto-scaled detector images without saving the figure into the file.

        '''
        fig, axs = plt.subplots(2, 1, figsize=(10,5))
        axs[0].imshow(self.RawImageA, cmap='Greys')
        axs[1].imshow(self.RawImageB, cmap='Greys')
        ttls = ['Detector A: '+self.DetA.strip('\x00'), 'Detector B: '+self.DetB.strip('\x00')]
        for ax, ttl in zip(axs, ttls):
            ax.axis(False)
            ax.set_title(ttl, fontsize=10)
        fig.suptitle(self.fname)
        
    def save_images_jpeg(self, **kwargs):
        '''
        Display auto-scaled detector images and save the figure into JPEG file (s).

        Parameters
        ----------
        kwargs:
        images_to_save : str
            Images to save. options are: 'A', 'B', or 'Both' (default).
        invert_data : boolean
            If True, the image will be inverted.
        
        '''
        images_to_save = kwargs.get("images_to_save", 'Both')
        invert_data = kwargs.get("invert_data", False)

        if images_to_save == 'Both' or images_to_save == 'A':
            if self.ftype == 0:
                fname_jpg = os.path.splitext(self.fname)[0] + '_' + self.DetA.strip('\x00') + '.jpg'
            else:
                fname_jpg = os.path.splitext(self.fname)[0] + 'DetA.jpg'
            Img = self.RawImageA_8bit_thresholds()[0]
            if invert_data:
                Img =  np.uint8(255) - Img
            PILImage.fromarray(Img).save(fname_jpg)

        try:
            if images_to_save == 'Both' or images_to_save == 'B':
                if self.ftype == 0:
                    fname_jpg = os.path.splitext(self.fname)[0] +  '_' + self.DetB.strip('\x00') + '.jpg'
                else:
                    fname_jpg = os.path.splitext(self.fname)[0] + 'DetB.jpg'
                Img = self.RawImageB_8bit_thresholds()[0]
                if invert_data:
                    Img =  np.uint8(255) - Img
                PILImage.fromarray(Img).save(fname_jpg)
        except:
            print('No Detector B image to save')

    def save_images_tif(self, images_to_save = 'Both'):
        '''
        Save the detector images into TIF file (s).

        Parameters
        ----------
        images_to_save : str
            Images to save. options are: 'A', 'B', or 'Both' (default).
        
        '''
        if self.ftype == 0:
            if images_to_save == 'Both' or images_to_save == 'A':
                fnameA = os.path.splitext(self.fname)[0] + '_' + self.DetA.strip('\x00') + '.tif'
                try:
                    tiff.imsave(fnameA, self.RawImageA)
                except:
                    tiff.imwrite(fnameA, self.RawImageA)
            if self.DetB != 'None':
                if images_to_save == 'Both' or images_to_save == 'B':
                    fnameB = os.path.splitext(self.fname)[0] + '_' + self.DetB.strip('\x00') + '.tif'
                    try:
                        tiff.imsave(fnameB, self.RawImageB)
                    except:
                        tiff.imwrite(fnameB, self.RawImageB)
        else:
            print('original File is already in TIF format')
        
    def get_image_min_max(self, image_name = 'ImageA', thr_min = 1.0e-4, thr_max = 1.0e-3, nbins=256, disp_res = False):
        '''
        Calculates the data range of the EM data. ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Calculates histogram of pixel intensities of the loaded image
        with number of bins determined by parameter nbins (default = 256)
        and normalizes it to get the probability distribution function (PDF),
        from which a cumulative distribution function (CDF) is calculated.
        Then given the thr_min, thr_max parameters,
        the minimum and maximum values for the image are found by finding
        the intensities at which CDF= thr_min and (1- thr_max), respectively.

        Parameters
        ----------
        image_name : string
            the name of the image to perform this operations (defaulut is 'RawImageA')
        thr_min : float
            CDF threshold for determining the minimum data value
        thr_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        disp_res : boolean
            (default is False) - to plot/ display the results

        Returns:
            dmin, dmax: (float) minimum and maximum values of the data range.   
        '''

        if image_name == 'ImageA':
            if not hasattr(self, 'ImageA'):
                self.calculate_scaled_images()
            im = self.ImageA
        if image_name == 'ImageB':
            if not hasattr(self, 'ImageB'):
                self.calculate_scaled_images()
            im = self.ImageB
        if image_name == 'RawImageA':
            im = self.RawImageA
        if image_name == 'RawImageB':
            im = self.RawImageB
        return get_min_max_thresholds(im, thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=disp_res)

    def RawImageA_8bit_thresholds(self, thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        '''
        Convert the Image A into 8-bit array

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF
        
        Returns
        dt, data_min, data_max
            dt : 2D np.uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        '''
        if self.EightBit==1:
            #print('8-bit image already - no need to convert')
            dt = self.RawImageA
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(image_name ='RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            dt = ((np.clip(self.RawImageA, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        return dt, data_min, data_max

    def RawImageB_8bit_thresholds(self, thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = -1, data_max = -1, nbins=256):
        '''
        Convert the Image B into 8-bit array

        Parameters
        ----------
        thr_min : float
            lower CDF threshold for determining the minimum data value
        thr_max : float
            upper CDF threshold for determining the maximum data value
        data_min : float
            If different from data_max, this value will be used as low bound for I8 data conversion
        data_max : float
            If different from data_min, this value will be used as high bound for I8 data conversion
        nbins : int
            number of histogram bins for building the PDF and CDF
        
        Returns
        dt, data_min, data_max
            dt : 2D np.uint8 array
                Converted data
            data_min : float
                value used as low bound for I8 data conversion
            data_max : float
                value used as high bound for I8 data conversion
        '''
        if self.EightBit==1:
            #print('8-bit image already - no need to convert')
            dt = self.RawImageB
        else:
            if data_min == data_max:
                data_min, data_max = self.get_image_min_max(image_name ='RawImageB', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
            dt = ((np.clip(self.RawImageB, data_min, data_max) - data_min)/(data_max-data_min)*255.0).astype(np.uint8)
        return dt, data_min, data_max
    
    def save_snapshot(self, **kwargs):
        '''
        Build an image that contains both the Detector A and Detector B (if present) images as well as a table with important FIB-SEM parameters.

        kwargs:
         ----------
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
        disp_res : boolean
            If True (default) display the results.
        dpi : int
            DPI. Default is 300.
        snapshot_name : string
            The name of the image to perform these operations (default is frame_name + '_snapshot.png').

        '''
        thr_min = kwargs.get('thr_min', 1.0e-3)
        thr_max = kwargs.get('thr_max', 1.0e-3)
        nbins = kwargs.get('nbins', 256)
        disp_res = kwargs.get('disp_res', True)
        dpi = kwargs.get('dpi', 300)
        snapshot_name = kwargs.get('snapshot_name', os.path.splitext(self.fname)[0] + '_snapshot.png')

        ifDetB = (self.DetB != 'None')
        if ifDetB:
            try:
                dminB, dmaxB = self.get_image_min_max(image_name ='RawImageB', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
                fig, axs = plt.subplots(3, 1, figsize=(11,8))
            except:
                ifDetB = False
                pass
        if not ifDetB:
            fig, axs = plt.subplots(2, 1, figsize=(7,8))
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.90, wspace=0.15, hspace=0.1)
        dminA, dmaxA = self.get_image_min_max(image_name ='RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins, disp_res=False)
        axs[1].imshow(self.RawImageA, cmap='Greys', vmin=dminA, vmax=dmaxA)
        if ifDetB:
            axs[2].imshow(self.RawImageB, cmap='Greys', vmin=dminB, vmax=dmaxB)
        try:
            ttls = [self.Notes.strip('\x00'),
                'Detector A:  '+ self.DetA.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(dminA, dmaxA, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessA, self.ContrastA),
                'Detector B:  '+ self.DetB.strip('\x00') + ',  Data Range:  {:.1f} ÷ {:.1f} with thr_min={:.1e}, thr_max={:.1e}'.format(dminB, dmaxB, thr_min, thr_max) + '    (Brightness: {:.1f}, Contrast: {:.1f})'.format(self.BrightnessB, self.ContrastB)]
        except:
            ttls = ['', 'Detector A', '']
        for j, ax in enumerate(axs):
            ax.axis(False)
            ax.set_title(ttls[j], fontsize=10)
        fig.suptitle(self.fname)
        
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
                          'Frame Size', '{:d} x {:d}'.format(self.XResolution, self.YResolution), '',
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
                              'Frame Size', '{:d} x {:d}'.format(self.XResolution, self.YResolution), '',
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
                              'Frame Size', '{:d} x {:d}'.format(self.XResolution, self.YResolution), '',
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
        tbl = axs[0].table(cellText=cell_text,
                           colWidths=clw,
                           cellLoc='center',
                           colLoc='center',
                           bbox = [0.02, 0, 0.96, 1.0],
                           #bbox = [0.45, 1.02, 2.8, 0.55],
                           zorder=10)

        fig.savefig(snapshot_name, dpi=dpi)
        if disp_res == False:
            plt.close(fig)
    
    def analyze_noise_ROIs(self, Noise_ROIs, Hist_ROI, **kwargs):
        '''
        Analyses the noise statistics in the selected ROI's of the EM data.
        (Calls Single_Image_Noise_ROIs(img, Noise_ROIs, Hist_ROI, **kwargs):).
        This function is somewhat obsolete. When possible – use a method analyze_noise_statistics(**kwargs):,
        which calls Single_Image_Noise_Statistics(img, **kwargs):.
        That is a more powerful and accurate way of analysing the noise distribution.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com

        Performs following:
        1. For each of the selected Noise_ROIs, this method will perfrom the following:
            1a. Smooth the data by 2D convolution with a given kernel.
            1b. Determine "Noise" as difference between the original raw and smoothed data.        
            1c. Calculate the mean intensity value of the data and variance of the above "Noise"
        2. Plot the dependence of the noise variance vs. image intensity.
        3. One of the parameters is a DarkCount. If it is not explicitly defined as input parameter,
            it will be determined from the header data:
                for RawImageA it is self.Scaling[1,0]
                for RawImageB it is self.Scaling[1,1]
        4. The liner equation is determined for a line that passes through the point:
                Intensity=DarkCount and Noise Variance = 0
                and is a best fit for the [Mean Intensity, Noise Variance] points
                determined for each ROI (Step 1 above).
        5. Another ROI (defined by Hist_ROI parameter) is used to built an
            intensity histogram of the actual data. Peak of that histogram is determined.
        6. The data is plotted. Two values of SNR are defined from the slope of the line in Step 4:
            PSNR (Peak SNR) = Mean Intensity/sqrt(Noise Variance) at the intensity
                at the histogram peak determined in the Step 5.
            DSNR (Dynamic SNR) = (Max Intensity - Min Intensity) / sqrt(Noise Variance),
                where Max and Min Intensity are determined by corresponding cummulative
                threshold parameters, and Noise Variance is taken at the intensity
                in the middle of the range (Min Intensity + Max Intensity)/2.0

        Parameters:
        ----------
        Noise_ROIs : list of lists: [[left, right, top, bottom]]
            List of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the noise.
        Hist_ROI : list [left, right, top, bottom]
            Coordinates (indices) of the boundaries of the image subset to evaluate the real data histogram.
        kwargs:
        ----------
        image_name : string
            The name of the image to perform this operations (default is 'RawImageA').
        DarkCount : float
            The value of the Intensity Data at 0.
        kernel : 2D float array
            A kernel to perform 2D smoothing convolution. Default is normalized np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]), where st = 1/np.sqrt(2).
        nbins_disp : int
            Number of histogram bins for building the PDF and CDF to determine the data range for data display. Default is 256.
        thresholds_disp : list [thr_min_disp, thr_max_disp]
            CDF threshold for determining the min and max data values for display. Default id [1e-3, 1e-3].
        nbins_analysis : int
            Number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5. Default is 256.
        thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
            CDF threshold for building the data histogram in Step 5. Default is [2e-2, 2e-2].
        disp_res : boolean
            If True - plot/ display the results. Default is True.
        save_res_png : boolean
            Save the analysis output into a PNG file. Default is True.
        res_fname : str
            Filename - used for plotting the data. Default is attribute self.fname  + '_' + image_name + '_Noise_Analysis_ROIs.png'.
        Returns:
        ----------
        mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR
            mean_vals and var_vals are the Mean Intensity and Noise Variance values for the Noise_ROIs (Step 1)
            NF_slope is the slope of the linear fit curve (Step 4)
            PSNR and DSNR are Peak and Dynamic SNR's (Step 6)
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        res_fname_default = os.path.splitext(self.fname)[0] + '_' + image_name + '_Noise_Analysis_ROIs.png'
        res_fname = kwargs.get("res_fname", res_fname_default)
        disp_res = kwargs.get('disp_res', True)
        save_res_png = kwargs.get('save_res_png', True)

        if image_name == 'RawImageA':
            ImgEM = self.RawImageA.astype(float)
            DarkCount = self.Scaling[1,0]
        if image_name == 'RawImageB' and self.DetB != 'None':
            ImgEM = self.RawImageB.astype(float)
            DarkCount = self.Scaling[1,1]

        if (image_name == 'RawImageA') or (image_name == 'RawImageB' and self.DetB != 'None'):
            st = 1.0/np.sqrt(2.0)
            def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
            def_kernel = def_kernel/def_kernel.sum()
            kernel = kwargs.get("kernel", def_kernel)
            DarkCount = kwargs.get("DarkCount", DarkCount)
            nbins_disp = kwargs.get("nbins_disp", 256)
            thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
            nbins_analysis = kwargs.get("nbins_analysis", 100)
            thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
            try:
                Notes = kwargs.get("Notes", self.Notes.strip('\x00'))
            except:
                Notes = ''
                pass
            kwargs['kernel'] = kernel
            kwargs['DarkCount'] = DarkCount
            kwargs['img_label'] = image_name
            kwargs['Notes'] = Notes
            kwargs['disp_res'] = disp_res
            kwargs['save_res_png'] = save_res_png
            kwargs['res_fname'] = res_fname
            mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR = Single_Image_Noise_ROIs(ImgEM, Noise_ROIs, Hist_ROI, **kwargs)

        else:
            print('No valid image name selected')
            mean_vals = 0.0
            var_vals = 0.0
            NF_slope = 0.0
            PSNR = 0.0
            MSNR = 0.0
            DSNR = 0.0

        return mean_vals, var_vals, NF_slope, PSNR, MSNR, DSNR

    def analyze_noise_statistics(self, **kwargs):
        '''
        Analyses the noise statistics of the EM data image.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
        
        Calls Single_Image_Noise_Statistics(img, **kwargs)
        Performs following:
        1. Smooth the image by 2D convolution with a given kernel.
        2. Determine "Noise Image" as difference between the original raw and smoothed data.
        3. Select subsets of original, smoothed and noise images by selecting only elements where the filter_array (optional input 2D array) is True. Use calculate_gradent_map(img, ** kwargs) to define filter_array.
        4. Build a histogram of Smoothed Image (subset if filter_array was set).
        5. For all pixels withing each histogram bin of the Smoothed Image from Step 4, calculate the mean value (mean of intensities of the select pixels of the Smoothed Image from Step 4) and variance (variance of intensities of the select pixels of the Noise Image from the Step 2).
        6. Plot the dependence of the image variance vs. image intensity.
        7. Perform free linear fit of the variance vs. intensity. SNR0 is calculated as <S^2>/<N^2>.
        8. Perform linear fit with forced zero Intercept (DarkCount) of the variance vs. intensity. SNR1 is calculated <S^2>/<N^2>.

        Parameters
        ----------
        kwargs:
        ----------
        image_name : string
            The name of the image to perform these operations. Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image noise statistics.
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        DarkCount : float
            The value of the Intensity Data at 0.
        filter_array : 2d boolean array
            Array of the same dimensions as image. Only the pixel with corresponding filter_array values of True will be considered in the noise analysis.
        kernel : 2D float array
            A kernel to perform 2D smoothing convolution. Default is normalized np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]), where st = 1/np.sqrt(2).
        nbins_disp : int
            Number of histogram bins for building the PDF and CDF to determine the data range for data display. Default is 256.
        thresholds_disp : list [thr_min_disp, thr_max_disp]
            CDF threshold for determining the min and max data values for display. Default id [1e-3, 1e-3].
        nbins_analysis : int
            Number of histogram bins for building the PDF and CDF to determine the data range for building the data histogram in Step 5. Default is 256.
        thresholds_analysis: list [thr_min_analysis, thr_max_analysis]
            CDF threshold for building the data histogram in Step 5. Default is [2e-2, 2e-2].
        disp_res : boolean
            If True - plot/ display the results. Default is True.
        disp_res_SNR0 : boolean
            Add SNR0 (free fit) results to Variance vs. Intensity plot. Default is True.
        disp_res_SNR1 : boolean
            Add SNR1 (use Dark Count from Scaling data) results to Variance vs. Intensity plot. Default is True.
        save_res_png : boolean
            Save the analysis output into a PNG file. Default is True.
        res_fname : str
            Filename - used for plotting the data. If not explicitly defined will use the instance attribute self.fname + '_Noise_Analysis_' + image_name + '.png'
        img_label : string
            Optional image label. Defaults is self.Sample_ID.
        Notes : string
            Optional additional notes. Defaults is self.Notes.
        dpi : int
            Resolution (DPI ) of the PNG image.

        Returns:
        mean_vals, var_vals, I0, SNR0, SNR1, popt, result
            mean_vals and var_vals are the Mean Intensity and Noise Variance values for Step 5
            I0 is zero intercept (should be close to DarkCount),
            SNR0, SNR1 are Peak and Dynamic SNR's (Step 7 and 8)
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        res_fname_default = os.path.splitext(self.fname)[0] + '_Noise_Analysis_' + image_name + '.png'
        res_fname = kwargs.get("res_fname", res_fname_default)

        if image_name == 'RawImageA':
            ImgEM = self.RawImageA.astype(float)
            DarkCount = self.Scaling[1,0]
        if image_name == 'RawImageB' and self.DetB != 'None':
            ImgEM = self.RawImageB.astype(float)
            DarkCount = self.Scaling[1,1]

        if (image_name == 'RawImageA') or (image_name == 'RawImageB' and self.DetB != 'None'):
            st = 1.0/np.sqrt(2.0)
            evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
            def_kernel = np.array([[st, 1.0, st],[1.0,1.0,1.0], [st, 1.0, st]]).astype(float)
            def_kernel = def_kernel/def_kernel.sum()
            kernel = kwargs.get("kernel", def_kernel)
            filter_array = kwargs.get('filter_array', (ImgEM*0+1)>0)
            DarkCount = kwargs.get("DarkCount", DarkCount)
            nbins_disp = kwargs.get("nbins_disp", 256)
            thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
            nbins_analysis = kwargs.get("nbins_analysis", 100)
            thresholds_analysis = kwargs.get("thresholds_analysis", [2e-2, 1e-2])
            disp_res = kwargs.get("disp_res", True)
            disp_res_SNR0 = kwargs.get("disp_res_SNR0", True)
            disp_res_SNR1 = kwargs.get("disp_res_SNR1", True)
            save_res_png = kwargs.get("save_res_png", True)
            
            try:
                img_label = kwargs.get("img_label", self.Sample_ID)
            except:
                img_label = ''
                pass
            try:
                Notes = kwargs.get("Notes", self.Notes.strip('\x00'))
            except:
                Notes = ''
                pass
            dpi = kwargs.get("dpi", 300)

            noise_kwargs = {'image_name' : image_name,
                            'evaluation_box' : evaluation_box,
                            'kernel' : kernel,
                            'filter_array' : filter_array,
                            'DarkCount' : DarkCount,
                            'nbins_disp' : nbins_disp,
                            'thresholds_disp' : thresholds_disp,
                            'nbins_analysis' : nbins_analysis,
                            'thresholds_analysis' : thresholds_analysis,
                            'disp_res' : disp_res,
                            'disp_res_SNR0' : disp_res_SNR0,
                            'disp_res_SNR1' : disp_res_SNR1,
                            'save_res_png' : save_res_png,
                            'res_fname' : res_fname,
                            'Notes' : Notes,
                            'dpi' : dpi}

            mean_vals, var_vals, I0, SNR0, SNR1, popt, result =  Single_Image_Noise_Statistics(ImgEM, **noise_kwargs)
        else:
            mean_vals, var_vals, I0, SNR0, SNR1, popt, result = [], [], 0.0, 0.0, 0.0, np.array((0.0, 0.0)), [] 
        return mean_vals, var_vals, I0, SNR0, SNR1, popt, result
    

    def analyze_SNR_autocorr(self, **kwargs):
        '''
        Estimates SNR using auto-correlation analysis of a single image.
        ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
        
        Calculates SNR of a single image base on auto-correlation analysis of a single image, after [1].
        Calls function Single_Image_SNR(img, **kwargs)
        
        Parameters:
        ---------
        kwargs:
        ---------
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'.
        zero_mean: boolean
            If True (default), auto-correlation is zero-mean.
        edge_fraction : float
            Fraction of the full auto-correlation  range used to calculate the "zero value" (default is 0.10).
        extrapolate_signal : str
            Extrapolation method to find signal auto-correlation at 0-point (“Noise-free Autocorrelation”). 
            Options are (default is 'parabolic'):
                'nearest'  - nearest point (1 pixel away from center)
                'linear'   - linear interpolation of 2-points next to center
                'parabolic' - parabolic interpolation of 2 points left and 2 points right 
                'gaussian'  - gaussian interpolation with number of points = aperture
                'LDR' - use Levinson-Durbin recusrsion (ACLDR in [1]).
        nlags : int
            In case of 'LDR' (Levinson-Durbin recursion) nlags is the recursion order (a number of lags). Default is min(xsize/4, ysize/4).
        aperture : int
            Total number of points for gaussian interpolation. Default is 6.
        evaluation_box : list of 4 int
            Evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
        disp_res : boolean
            Display results (plots) (default is True).
        save_res_png : boolean
            Save the analysis output into a PNG file (default is True).
        res_fname : string
            Filename for the result image. Default is self.fname + '_AutoCorr_Noise_Analysis_' + image_name + '.png'
        img_label : string
            Optional image label.
        dpi : int
            Resolution (DPI) for the output PNG image.
        verbose : boolean
            Display intermediate results. Default is False.
        Returns:
        ---------
            xSNR, ySNR : float, float
                SNR determined using the method in [1] along X- and Y- directions.
                If there is a direction with slow varying data - that direction provides more accurate SNR estimate.
                Y-streaks in typical FIB-SEM data provide slow varying Y-component because streaks usually get increasingly worse with increasing Y.
                So for typical FIB-SEM data use ySNR or rSNR.   
                [1] J. T. L. Thong et al, Single-image signal-tonoise ratio estimation. Scanning, 328–336 (2001).
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        zero_mean = kwargs.get('zero_mean', True)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        edge_fraction = kwargs.get("edge_fraction", 0.10)
        extrapolate_signal = kwargs.get('extrapolate_signal', 'parabolic')
        aperture = kwargs.get('aperture', 6)
        disp_res = kwargs.get("disp_res", True)
        save_res_png = kwargs.get("save_res_png", True)
        default_res_name = os.path.splitext(self.fname)[0] + '_AutoCorr_Noise_Analysis_' + image_name + '.png'
        res_fname = kwargs.get("res_fname", default_res_name)
        dpi = kwargs.get("dpi", 300)
        verbose = kwargs.get('verbose', False)

        if image_name == 'RawImageA':
            img = self.RawImageA
        if image_name == 'RawImageB':
            img = self.RawImageB
        if image_name == 'ImageA':
            if not hasattr(self, 'ImageA'):
                self.calculate_scaled_images()
            img = self.ImageA
        if image_name == 'ImageB':
            if not hasattr(self, 'ImageB'):
                self.calculate_scaled_images()
            img = self.ImageB
        ysz, xsz = img.shape
        nlags = kwargs.get("nlags", np.min((ysz, xsz))//4)

        SNR_kwargs = {'edge_fraction' : edge_fraction,
                        'zero_mean' : zero_mean,
                        'evaluation_box' : evaluation_box,
                        'extrapolate_signal' : extrapolate_signal,
                        'nlags' : nlags,
                        'aperture' : aperture,
                        'disp_res' : disp_res,
                        'save_res_png' : save_res_png,
                        'res_fname' : res_fname,
                        'img_label' : image_name,
                        'verbose' : verbose,
                        'dpi' : dpi}

        xi = 0
        yi = 0
        xa = xi + xsz
        ya = yi + ysz
        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya

        xSNR, ySNR, rSNR= Single_Image_SNR(img[yi_eval:ya_eval, xi_eval:xa_eval], **SNR_kwargs)

        return xSNR, ySNR, rSNR


    def show_eval_box(self, **kwargs):
        '''
        Show the box used for noise analysis.
        ©G.Shtengel, 04/2021. gleb.shtengel@gmail.com

        kwargs:
        ---------
        evaluation_box : list of 4 int
            Evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        data_dir : str
            Data directory (path).
        Sample_ID : str
            Sample ID
        box_linewidth : float
            Linewidth for the box outline. Default is 1.0.
        box_color : color
            Color for the box outline. Default is yellow.
        invert_data : boolean
            If True - the data is inverted.
        save_res_png  : boolean
            Save PNG image of the frame overlaid with evaluation box.
        '''
        image_name = kwargs.get("image_name", 'RawImageA')
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0]) 
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", os.path.dirname(self.fname))
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        nbins_disp = kwargs.get("nbins_disp", 256)
        thresholds_disp = kwargs.get("thresholds_disp", [1e-3, 1e-3])
        box_linewidth = kwargs.get("box_linewidth", 1.0)
        box_color = kwargs.get("box_color", 'yellow')
        invert_data =  kwargs.get("invert_data", False)
        save_res_png  = kwargs.get("save_res_png", False )

        if image_name == 'RawImageA':
            img = self.RawImageA
        if image_name == 'RawImageB':
            img = self.RawImageB
        if image_name == 'ImageA':
            if not hasattr(self, 'ImageA'):
                self.calculate_scaled_images()
            img = self.ImageA
        if image_name == 'ImageB':
            if not hasattr(self, 'ImageB'):
                self.calculate_scaled_images()
            img = self.ImageB

        xi = 0
        yi = 0
        ysz, xsz = img.shape
        xa = xi + xsz
        ya = yi + ysz
        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya

        range_disp = get_min_max_thresholds(img[yi_eval:ya_eval, xi_eval:xa_eval], thr_min = thresholds_disp[0], thr_max = thresholds_disp[1], nbins = nbins_disp, disp_res=False)

        fig, ax = plt.subplots(1,1, figsize = (10.0, 11.0*ysz/xsz))
        ax.imshow(img, cmap='Greys', vmin = range_disp[0], vmax = range_disp[1])
        ax.grid(True, color = "cyan")
        ax.set_title(self.fname)
        rect_patch = patches.Rectangle((xi_eval,yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2,
            linewidth=box_linewidth, edgecolor=box_color, facecolor='none')
        ax.add_patch(rect_patch)
        if save_res_png :
            fig.savefig(os.path.splitext(self.fname)[0]+'_evaluation_box.png', dpi=300)


    def determine_field_fattening_parameters(self, **kwargs):
        '''
        Perfrom 2D polynomial fit (calls Perform_2D_fit(Img, estimator, **kwargs)) and determine the field-flattening parameters.
        ©G.Shtengel, 04/2023. gleb.shtengel@gmail.com
        
        Parameters:
        ----------
        kwargs:
        image_names : list of str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        estimator : RANSACRegressor(),
                    LinearRegression(),
                    TheilSenRegressor(),
                    HuberRegressor(),
                    RidgeCV(),
                    LassoCV()
        bins : int
            Binning size (in pixel units) for image binning. Default is 10.
        Analysis_ROIs : list of lists: [[left, right, top, bottom]]
            List of coordinates (indices) for each of the ROI's - the boundaries of the image subset to evaluate the parabolic fit.
        calc_corr : boolean
            If True - the full image correction is calculated. Default is False.
        degrees : [int] 
            The maximal degree of the polynomial features for sklearn.preprocessing.PolynomialFeatures. 
            should be a list of the same length as image_names. If a single value, then it is applied to all images. Default is 2.
        ignore_Y  : boolean
            If True - the polynomial fit to only X is performed. Default is False.
        linear_Y  : boolean
            If True - the polynomial fit to only X is perfromed, only linear variation along Y is allowed.
        Xsect : int
            X - coordinate for Y-cross-section.
        Ysect : int
            Y - coordinate for X-cross-section.
        disp_res : boolean
            Plot/ display the results. Defaults is True.
        save_res_png : boolean
            Save the analysis output into a PNG file. Default is False.
        save_correction_binary : boolean
            Save the image_name and img_correction_array data into a binary file. Default is False.
        res_fname : string
            Filename for the result image ('**_Image_Flattening.png'). Default is derived from the same root, e.g. '**_Image_Flattening.bin'.
        label : string
            Optional image label.
        dpi : int
            Resolution (DPI) for the output PNG image.

        Returns:
        img_correction_coeffs, img_correction_arrays
        '''
        image_names = kwargs.get("image_names", ['RawImageA'])
        estimator = kwargs.get("estimator", LinearRegression())
        if "estimator" in kwargs:
            del kwargs["estimator"]
        calc_corr = kwargs.get("calc_corr", False)
        ignore_Y = kwargs.get("ignore_Y", False)
        linear_Y = kwargs.get('linear_Y', False)
        lbl = kwargs.get("label", '')
        disp_res = kwargs.get("disp_res", True)
        bins = kwargs.get("bins", 10) #bins = 10
        Analysis_ROIs = kwargs.get("Analysis_ROIs", [])
        save_res_png = kwargs.get("save_res_png", False)
        res_fname = kwargs.get("res_fname", os.path.splitext(self.fname)[0]+'_Image_Flattening.png')
        save_correction_binary = kwargs.get("save_correction_binary", False)
        dpi = kwargs.get("dpi", 300)
        degrees = kwargs.get('degree', 2)

        img_correction_arrays = []
        img_correction_coeffs = []
        img_correction_intercepts = []
        for j, image_name in enumerate(image_names):
            if image_name == 'RawImageA':
                img = self.RawImageA - self.Scaling[1,0]
            if image_name == 'RawImageB':
                img = self.RawImageB - self.Scaling[1,1]
            if image_name == 'ImageA':
                if not hasattr(self, 'ImageA'):
                    self.calculate_scaled_images()
                img = self.ImageA
            if image_name == 'ImageB':
                if not hasattr(self, 'ImageB'):
                    self.calculate_scaled_images()
                img = self.ImageB

            ysz, xsz = img.shape
            Xsect = kwargs.get("Xsect", xsz//2)
            Ysect = kwargs.get("Ysect", ysz//2)

            Fit_kwargs = {'image_name' : image_name,
                        'calc_corr' : calc_corr,
                        'ignore_Y' : ignore_Y,
                        'linear_Y' : linear_Y,
                        'Xsect' : Xsect,
                        'Ysect' : Ysect,
                        'disp_res' : disp_res,
                        'bins' : bins,
                        'Analysis_ROIs' : Analysis_ROIs,
                        'save_res_png' : save_res_png,
                        'res_fname' : res_fname.replace('.png', '_' + image_name + '.png'),
                        'dpi' : dpi}
            try:
                Fit_kwargs['degree'] = degrees[j]
            except:
                Fit_kwargs['degree'] = 2
                pass
            intercept, coefs, mse, img_correction_array = Perform_2D_fit(img, estimator, **Fit_kwargs)
            img_correction_arrays.append(img_correction_array)
            img_correction_coeffs.append(coefs)
            img_correction_intercepts.append(intercept)
        kwargs['res_fname'] = res_fname

        if calc_corr:
            self.image_correction_sources = image_names
            self.img_correction_arrays = img_correction_arrays
            if save_correction_binary:
                bin_fname = res_fname.replace('png', 'bin')
                pickle.dump([image_names, img_correction_arrays], open(bin_fname, 'wb')) # saves source name and correction array into the binary file
                self.image_correction_file = res_fname.replace('png', 'bin')
                print('Image Flattening Info saved into the binary file: ', self.image_correction_file)
        #self.intercept = intercept
        self.img_correction_coeffs = img_correction_coeffs
        return img_correction_intercepts, img_correction_coeffs, img_correction_arrays

        
    def flatten_image(self, **kwargs):
        '''
        Flatten the image(s). Image flattening parameters must be determined (determine_field_fattening_parameters)
        ©G.Shtengel, 04/2023. gleb.shtengel@gmail.com
        Parameters:
        ----------
        kwargs:
        ----------
        image_correction_file : str
            Full path to a binary filename that contains source names (image_correction_sources) and correction arrays (img_correction_arrays).
            If image_correction_file exists, the data is loaded from it.
        image_correction_sources : list of str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
        img_correction_arrays : list of 2D arrays
            Arrays containing field flattening info.

        Returns:
        flattened_images : list of 2D arrays
        '''

        if hasattr(self, 'image_correction_file'):
            image_correction_file = kwargs.get("image_correction_file", self.image_correction_file)
        else:
            image_correction_file = kwargs.get("image_correction_file", '')

        try:
            # try loading the image correction data from the binary file
            with open(image_correction_file, "rb") as f:
                [image_correction_sources,  img_correction_arrays] = pickle.load(f)
        except:
            #  if that did not work, see if the correction data was provided directly
            if hasattr(self, 'image_correction_source'):
                image_correction_sources = kwargs.get("image_correction_sources", self.image_correction_sources)
            else:
                image_correction_sources = kwargs.get("image_correction_sources", [False])

            if hasattr(self, 'img_correction_arrays'):
                img_correction_arrays = kwargs.get("img_correction_arrays", self.img_correction_arrays)
            else:
                img_correction_arrays = kwargs.get("img_correction_arrays", [False])

        calculate_scaled_images = (('ImageA' in image_correction_sources) and (not hasattr(self, 'ImageA'))) or (('ImageB' in image_correction_sources) and (not hasattr(self, 'ImageB')))
        if calculate_scaled_images:
            self.calculate_scaled_images()

        flattened_images = []
        for image_correction_source, img_correction_array in zip(image_correction_sources, img_correction_arrays):
            if (image_correction_source is not False) and (img_correction_array is not False):
                if image_correction_source == 'RawImageA':
                    flattened_image = (self.RawImageA - self.Scaling[1,0])*img_correction_array[0:self.YResolution, 0:self.XResolution] + self.Scaling[1,0]
                if image_correction_source == 'RawImageB':
                    flattened_image = (self.RawImageB - self.Scaling[1,1])*img_correction_array[0:self.YResolution, 0:self.XResolution] + self.Scaling[1,1]
                if image_correction_source == 'ImageA':                
                    flattened_image = self.ImageA*img_correction_array[0:self.YResolution, 0:self.XResolution]
                if image_correction_source == 'ImageB':
                    flattened_image = self.ImageB*img_correction_array[0:self.YResolution, 0:self.XResolution]
            else:
                if image_correction_source == 'RawImageA':
                    flattened_image = self.RawImageA
                if image_correction_source == 'RawImageB':
                    flattened_image = self.RawImageB
                if image_correction_source == 'ImageA':
                    flattened_image = self.ImageA
                if image_correction_source == 'ImageB':
                    flattened_image = self.ImageB
            flattened_images.append(flattened_image)

        return flattened_images


###################################################
#   Helper functions for FIBSEM_dataset class
###################################################

def kp_to_list(kp):
    '''
    Convert a keypont object to a list (so that it can be "pickled").
    
    Returns
    pt, angle, size, response, class_id, octave
    (all extracted from corresponding cv2.KeyPoint() object attributes)
    '''
    x, y = kp.pt
    pt = float(x), float(y)
    angle = float(kp.angle) if kp.angle is not None else None
    size = float(kp.size) if kp.size is not None else None
    response = float(kp.response) if kp.response is not None else None
    class_id = int(kp.class_id) if kp.class_id is not None else None
    octave = int(kp.octave) if kp.octave is not None else None
    return pt, angle, size, response, class_id, octave

def list_to_kp(inp_list):
    '''
    Convert a list to a keypont object

    Parameters:
    inp_list : list
        List of Key-Point properties to initialize the following cv2.KeyPoint() object attributes:
        [pt, angle, size, response, class_id, octave]

    Returns:
    kp : Instance of cv2.KeyPoint() object 
    '''
    kp = cv2.KeyPoint()
    kp.pt = inp_list[0]
    kp.angle = inp_list[1]
    kp.size = inp_list[2]
    kp.response = inp_list[3]
    kp.octave = inp_list[4]
    kp.class_id = inp_list[5]
    return kp


def evaluate_FIBSEM_frame(params):
    '''
    Evaluates single FIB-SEM frame and extract parameters: data min/max, milling rate, FOV center.
    1. Calculates the data range of the EM data ©G.Shtengel 04/2022 gleb.shtengel@gmail.com
    Calculates histogram of pixel intensities of the loaded image
    with number of bins determined by parameter nbins (default = 256)
    and normalizes it to get the probability distribution function (PDF),
    from which a cumulative distribution function (CDF) is calculated.
    Then given the thr_min, thr_max parameters,
    the minimum and maximum values for the image are found by finding
    the intensities at which CDF= thr_min and (1- thr_max), respectively.
    2. Extracts WD, MillingYVoltage, center_x, center_y data from the header
    
    Parameters:
    ----------
    params =  fl, kwargs
        fl : str
            The string containing a full path to the EM data file.
        kwargs: dictioanry of kwargs:
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        image_name: string
            the name of the image to perform this operations (defaulut is 'RawImageA')
        thr_min : float
            CDF threshold for determining the minimum data value
        thr_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF

    Returns:
        dmin, dmax: (float) minimum and maximum values of the data range.
        WD, MillingYVoltage, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolution, YResolution - SEM parameters 
    '''
    fl, kwargs = params
    ftype = kwargs.get("ftype", 0)
    image_name = kwargs.get("image_name", 'RawImageA')
    calculate_scaled_images = (image_name == 'ImageA') or (image_name == 'ImageB')
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    ex_error=None
    try:
        frame = FIBSEM_frame(fl, ftype=ftype, calculate_scaled_images=calculate_scaled_images)
        if frame.EightBit ==1:
            dmin = np.uint8(0)
            dmax =  np.uint8(255)
        else:
            dmin, dmax = frame.get_image_min_max(image_name = image_name, thr_min=thr_min, thr_max=thr_max, nbins=nbins)
        if ftype == 0:
            try:
                WD = frame.WD
            except:
                WD = 0
            try:
                MillingYVoltage = frame.MillingYVoltage
            except:
                MillingYVoltage = 0
            try:
                ScanRate = frame.ScanRate
            except:
                ScanRate = 0
            try:
                EHT = frame.EHT
            except:
                EHT = 0
            try:
                SEMSpecimenI = -1.0* frame.SEMSpecimenI
            except:
                SEMSpecimenI = 0
            try:
                center_x = (frame.FirstPixelX + frame.XResolution/2.0)
            except:
                center_x = 0
            try:
                center_y = (frame.FirstPixelY + frame.YResolution/2.0)
            except:
                center_y = 0
            try:
                XResolution = frame.XResolution
            except:
                XResolution = FIBSEM_frame.RawImageA.shape[1]
            try:
                YResolution = frame.YResolution
            except:
                YResolution = FIBSEM_frame.RawImageA.shape[1]
        else:
            WD = 0
            MillingYVoltage = 0
            center_x = 0
            center_y = 0
            ScanRate = 0
            EHT = 0
            SEMSpecimenI = 0
            XResolution = 0
            YResolution = 0
    except Exception as err:
        dmin = 0
        dmax = 0
        WD = 0
        MillingYVoltage = 0
        center_x = 0
        center_y = 0
        ScanRate = 0
        EHT = 0
        SEMSpecimenI = 0
        ex_error = err
        XResolution = 0
        YResolution = 0

    return dmin, dmax, WD, MillingYVoltage, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolution, YResolution, ex_error


def evaluate_FIBSEM_frames_dataset(fls, DASK_client, **kwargs):
    '''
    Evaluates parameters of FIBSEM data set (Min/Max, Working Distance (WD), Milling Y Voltage (MV), FOV center positions).

    Parameters:
    DASK_client  : DASK client

    kwargs:
    use_DASK : boolean
        perform remote DASK computations
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    frame_inds : array
        Array of frames to be used for evaluation. If not provided, evaluzation will be performed on all frames
    data_dir : str
        data directory (path) for saving the data
    image_name: string
            the name of the image to perform this operations (defaulut is 'RawImageA')
    thr_min : float
        CDF threshold for determining the minimum data value
    thr_max : float
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    sliding_minmax : boolean
        if True - data min and max will be taken from data_min_sliding and data_max_sliding arrays
        if False - same data_min_glob and data_max_glob will be used for all files
    fit_params : list
        Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
        Other options are:
            ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
            ['PF', 2]  - use polynomial fit (in this case of order 2)
    Mill_Volt_Rate_um_per_V : float
        Milling Voltage to Z conversion (µm/V). Default is 31.235258870176065.
    FIBSEM_Data_xlsx : str
        Filepath of the Excell file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions)
    disp_res : boolean
        If True (default), intermediate messages and results will be displayed.
    use_existing_data : boolean
        Default is False. If True and the data exists (saved into XLSX), use that.   

    Returns:
    list of 14 parameters: FIBSEM_Data_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions
        FIBSEM_Data_xlsx : str
            path to Excel file with the FIBSEM data
        data_min_glob : float   
            min data value for I8 conversion (open CV SIFT requires I8)
        data_man_glob : float   
            max data value for I8 conversion (open CV SIFT requires I8)
        data_min_sliding : float array
            min data values (one per file) for I8 conversion
        data_max_sliding : float array
            max data values (one per file) for I8 conversion
        mill_rate_WD : float array
            Milling rate calculated based on Working Distance (WD)
        mill_rate_MV : float array
            Milling rate calculated based on Milling Y Voltage (MV)
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

    nfrs = len(fls)
    use_DASK = kwargs.get("use_DASK", False)
    DASK_client_retries = kwargs.get("DASK_client_retries", 3)
    ftype = kwargs.get("ftype", 0)
    frame_inds = kwargs.get("frame_inds", np.arange(len(fls)))
    data_dir = kwargs.get("data_dir", '')
    image_name = kwargs.get("image_name", 'RawImageA')
    calculate_scaled_images = (image_name == 'ImageA') or (image_name == 'ImageB')
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    sliding_minmax = kwargs.get("sliding_minmax", True)
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    if fit_params[0] != 'None':
        sv_apert = min([fit_params[1], len(frame_inds)//8*2+1])
    Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", 31.235258870176065)
    kwargs['Mill_Volt_Rate_um_per_V'] = Mill_Volt_Rate_um_per_V
    
    FIBSEM_Data_xlsx = kwargs.get('FIBSEM_data_xlsx', 'FIBSEM_Data.xlsx')
    FIBSEM_Data_xlsx_path = os.path.join(data_dir, FIBSEM_Data_xlsx)
    disp_res = kwargs.get("disp_res", False)
    use_existing_data = kwargs.get('use_existing_data', False)

    if use_existing_data and os.path.exists(FIBSEM_Data_xlsx_path):
        int_results = pd.read_excel(FIBSEM_Data_xlsx_path, sheet_name='FIBSEM Data')
        fr = int_results['Frame']
        data_min = int_results['Min']
        data_max = int_results['Max']
        data_minmax_glob = np.vstack((data_min, data_max)).T
        mill_rate_WD = int_results['Working Distance (mm)']
        mill_rate_MV = int_results['Milling Y Voltage (V)']
        center_x = int_results['FOV X Center (Pix)']
        center_y = int_results['FOV Y Center (Pix)']
        ScanRate = int_results['Scan Rate (Hz)']
        EHT = int_results['EHT (kV)']
        try:
            errors_s2 = int_results['Error Code']
        except:
            errors_s2 = fr * 0
        try:
            SEMSpecimenI = int_results['SEMSpecimenI (nA)']
        except:
            SEMSpecimenI = EHT*0.0
        try:
            XResolutions = int_results['XResolutions']
        except:
            XResolutions = fr * 0
        try:
            YResolutions = int_results['YResolutions']
        except:
            YResolutions = fr * 0
        data_min_glob, trash = get_min_max_thresholds(data_minmax_glob[:, 0], thr_min = thr_min, thr_max = thr_max, nbins = nbins, disp_res=False)
        trash, data_max_glob = get_min_max_thresholds(data_minmax_glob[:, 1], thr_min = thr_min, thr_max = thr_max, nbins = nbins, disp_res=False)
        if fit_params[0] != 'None':
            sv_apert = min([fit_params[1], len(frame_inds)//8*2+1])
            print('Using fit_params: ', 'SG', sv_apert, fit_params[2])
            data_min_sliding = savgol_filter(data_minmax_glob[:, 0].astype(np.double), sv_apert, fit_params[2])
            data_max_sliding = savgol_filter(data_minmax_glob[:, 1].astype(np.double), sv_apert, fit_params[2])
        else:
            print('Not smoothing the Min/Max data')
            data_min_sliding = data_minmax_glob[:, 0].astype(np.double)
            data_max_sliding = data_minmax_glob[:, 1].astype(np.double)
    else:
        frame = FIBSEM_frame(fls[0], ftype=ftype, calculate_scaled_images=calculate_scaled_images)
        if frame.EightBit == 1 and ftype == 1:
            if disp_res:
                print('Original data is 8-bit, no need to find Min and Max for 8-bit conversion')
            data_min_glob = np.uint8(0)
            data_max_glob =  np.uint8(255)
            data_min_sliding = np.zeros(nfrs, dtype=np.uint8)
            data_max_sliding = np.full(nfrs, np.uint8(255), dtype=np.uint8)
            data_minmax_glob = np.zeros((nfrs, 2), dtype=np.uint8)
            data_minmax_glob[1, :] = np.uint8(255)
            mill_rate_WD = np.zeros(nfrs, dtype=float)
            mill_rate_MV = np.zeros(nfrs, dtype=float)
            center_x = np.zeros(nfrs, dtype=float)
            center_y = np.zeros(nfrs, dtype=float)
            XResolutions = np.zeros(nfrs, dtype=int)
            YResolutions = np.zeros(nfrs, dtype=int)
            ScanRate = np.zeros(nfrs, dtype=float)
            EHT = np.zeros(nfrs, dtype=float)
            SEMSpecimenI = np.zeros(nfrs, dtype=float)
            errors_s2 = np.zeros(nfrs, dtype=int)

        else:
            params_s2 = [[fl, kwargs] for fl in np.array(fls)[frame_inds]]
            results_s2 = np.zeros((len(frame_inds), 11))
            errors_s2 = []
            if use_DASK:
                if disp_res:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
                futures = DASK_client.map(evaluate_FIBSEM_frame, params_s2, retries = DASK_client_retries)
                results_temp = np.array(DASK_client.gather(futures))
                for j, res_temp in enumerate(tqdm(results_temp, desc='Converting the Results', display = disp_res)):
                    results_s2[j, :] = res_temp[0:11]
                    errors_s2.append(res_temp[11])
            else:
                if disp_res:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using Local Computation')
                for j, param_s2 in enumerate(tqdm(params_s2, desc='Evaluating FIB-SEM frames (data min/max, mill rate, FOV shifts): ', display = disp_res)):
                    res_temp = evaluate_FIBSEM_frame(param_s2)
                    results_s2[j, :] = res_temp[0:11]
                    errors_s2.append(res_temp[11])

            data_minmax_glob = results_s2[:, 0:2]
            data_min_glob, trash = get_min_max_thresholds(data_minmax_glob[:, 0], thr_min = thr_min, thr_max = thr_max, nbins = nbins, disp_res=False)
            trash, data_max_glob = get_min_max_thresholds(data_minmax_glob[:, 1], thr_min = thr_min, thr_max = thr_max, nbins = nbins, disp_res=False)
            if fit_params[0] != 'None':
                sv_apert = min([fit_params[1], len(frame_inds)//8*2+1])
                print('Using fit_params: ', 'SG', sv_apert, fit_params[2])
                data_min_sliding = savgol_filter(data_minmax_glob[:, 0].astype(np.double), sv_apert, fit_params[2])
                data_max_sliding = savgol_filter(data_minmax_glob[:, 1].astype(np.double), sv_apert, fit_params[2])
            else:
                print('Not smoothing the Min/Max data')
                data_min_sliding = data_minmax_glob[:, 0].astype(np.double)
                data_max_sliding = data_minmax_glob[:, 1].astype(np.double)
            mill_rate_WD = results_s2[:, 2]
            mill_rate_MV = results_s2[:, 3]
            center_x = results_s2[:, 4]
            center_y = results_s2[:, 5]
            ScanRate = results_s2[:, 6]
            EHT = results_s2[:, 7]
            SEMSpecimenI = results_s2[:, 8]
            XResolutions = results_s2[:, 9].astype(int)
            YResolutions = results_s2[:, 10].astype(int)

    if disp_res:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the FIBSEM dataset statistics (Min/Max, Mill Rate, FOV Shifts into the file: ', FIBSEM_Data_xlsx_path)
        # Create a Pandas Excel writer using XlsxWriter as the engine.
    xlsx_writer = pd.ExcelWriter(FIBSEM_Data_xlsx_path, engine='xlsxwriter')
    columns=['Frame', 'Min', 'Max', 'Sliding Min', 'Sliding Max', 'Working Distance (mm)', 'Milling Y Voltage (V)', 'FOV X Center (Pix)', 'FOV Y Center (Pix)', 'XResolutions', 'YResolutions', 'Scan Rate (Hz)', 'EHT (kV)', 'SEMSpecimenI (nA)', 'Error Code']
    minmax_df = pd.DataFrame(np.vstack((frame_inds.T,
        data_minmax_glob.T,
        data_min_sliding.T,
        data_max_sliding.T,
        mill_rate_WD.T,
        mill_rate_MV.T,
        center_x.T,
        center_y.T,
        XResolutions.T,
        YResolutions.T,
        ScanRate.T,
        EHT.T,
        SEMSpecimenI.T,
        np.array(errors_s2).T)).T, columns = columns, index = None)
    minmax_df.to_excel(xlsx_writer, index=None, sheet_name='FIBSEM Data')
    kwargs_info = pd.DataFrame([kwargs]).T   # prepare to be save in transposed format
    kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')
    #xlsx_writer.save()
    xlsx_writer.close()
           
    return [FIBSEM_Data_xlsx_path, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions, errors_s2]


# Routines to extract Key-Points and Descriptors

def extract_keypoints_descr_files(params):
    '''
    Extracts Key-Points and Descriptors (single image) for SIFT procedure.
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    
    Parameters:
    -----------
    params = fl, dmin, dmax, kwargs
        fl : str
            image filename (full path)
        dmin : float   
            min data value for I8 conversion (open CV SIFT requires I8)
        dmax : float   
            max data value for I8 conversion (open CV SIFT requires I8)
        kwargs:
        -------
        ftype : int
            file type (0 - Shan Xu's .dat, 1 - tif)
        thr_min : float
            CDF threshold for determining the minimum data value
        thr_max : float
            CDF threshold for determining the maximum data value
        nbins : int
            number of histogram bins for building the PDF and CDF
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for key-point extraction.
        verbose : boolean
            If True, intermediate printouts are enabled. Default is False
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
            Note that the its meaning is different from the contrastThreshold,
            i.e. the larger the edgeThreshold, the less features are filtered out
            (more features are retained).
        SIFT_sigma : double
            SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
            If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.

    Returns:
        fnm : str
            path to the file containing Key-Points and Descriptors
    '''
    fl, dmin, dmax, kwargs = params
    ftype = kwargs.get("ftype", 0)
    verbose = kwargs.get("verbose", False)
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    use_existing_data = kwargs.get('use_existing_data', False)
    fnm = os.path.splitext(fl)[0] + '_kpdes.bin'

    if use_existing_data and os.path.exists(fnm):
        pass
    else:
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", 3)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.04)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 10)
        SIFT_sigma = kwargs.get("SIFT_sigma", 1.6)

        #sift = cv2.xfeatures2d.SIFT_create(nfeatures=SIFT_nfeatures, nOctaveLayers=SIFT_nOctaveLayers, edgeThreshold=SIFT_edgeThreshold, contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma)
        sift = cv2.SIFT_create(nfeatures=SIFT_nfeatures, nOctaveLayers=SIFT_nOctaveLayers, edgeThreshold=SIFT_edgeThreshold, contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma)
        img, d1, d2 = FIBSEM_frame(fl, ftype=ftype, calculate_scaled_images=False).RawImageA_8bit_thresholds(thr_min = 1.0e-3, thr_max = 1.0e-3, data_min = dmin, data_max = dmax, nbins=256)
        # extract keypoints and descriptors for both images

        xi_eval = evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = -1
        yi_eval = evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = -1

        kps, dess = sift.detectAndCompute(img[yi_eval:ya_eval, xi_eval:xa_eval], None)
        if xi_eval >0 or yi_eval>0:   # add shifts to ke-pint coordinates to convert them to full image coordinated
            for kp in kps:
                kp.pt = kp.pt + np.array((xi_eval, yi_eval))
        #key_points = [KeyPoint(kp) for kp in kps]
        if verbose:
            print('File: ', fl, ', extracted {:d} keypoints'.format(len(kps)))
        key_points = [kp_to_list(kp) for kp in kps]
        kpd = [key_points, dess]
        
        pickle.dump(kpd, open(fnm, 'wb')) # converts array to binary and writes to output
        #pickle.dump(dess, open(fnm, 'wb')) # converts array to binary and writes to output
    return fnm


def estimate_kpts_transform_error(src_pts, dst_pts, transform_matrix):
    """ 
    Estimate the transformation error for key-point pairs and known transformation matrix.
    ©G.Shtengel, 09/2021. gleb.shtengel@gmail.com

    Image transformation matrix in a form:
        A = [[a0  a1   a2]
             [b0   b1  b2]
             [0   0    1]]
     Thransofrmation is supposed to be in a form:
     Xnew = a0 * Xoriginal + a1 * Yoriginal + a2
     Ynew = b0 * Xoriginal + b1 * Yoriginal + b2
     source and destination points are pairs of coordinates (2xN array)
     errors are estimated as norm(dest_pts - A*src_pts) so that least square regression can be performed

    Returns:
        np.linalg.norm(dst_pts - src_pts_transformed, ord=2, axis=1), xshifts, yshifts
    """
    src_pts_transformed = src_pts @ transform_matrix[0:2, 0:2].T + transform_matrix[0:2, 2]
    xshifts = (dst_pts - src_pts_transformed)[:,0]
    yshifts = (dst_pts - src_pts_transformed)[:,1]
    return np.linalg.norm(dst_pts - src_pts_transformed, ord=2, axis=1), xshifts, yshifts


def determine_transformation_matrix(src_pts, dst_pts, **kwargs):
    '''
    Determine the transformation matrix.
    ©G.Shtengel, 09/2021. gleb.shtengel@gmail.com

    Determine the transformation matrix in a form:
            A = [[a0  a1   a2]
                 [b0   b1  b2]
                 [0   0    1]]
    based on the given source and destination points using linear regression such that the error is minimized for 
    sum(dst_pts - A*src_pts).
    
    For each matched pair of keypoins the error is calculated as err[j] = dst_pts[j] - A*src_pts[j]
    The iterative procedure throws away the matched keypoint pair with worst error on every iteration
    untill the worst error falls below drmax or the max number of iterations is reached.

    kwargs:
    TransformType - transformation type to be used (ShiftTransform, XScaleShiftTransform, ScaleShiftTransform, AffineTransform, RegularizedAffineTransform).
        Default is RegularizedAffineTransform.
    drmax : float
        In the case of 'LinReg' - outlier threshold for iterative regression
        In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
    max_iter : int
        Max number of iterations. Defaults is 1000
    remove_per_iter : int
        Number of worst outliers to remove per iteration. Defaults is 1.
    start : string
        'edges' (default) or 'center'. Start of search (registration error histogram evaluation).
    estimation : string
        'interval' (default) or 'count'. Returns a width of interval determined using search direction from above or total number of bins above half max (registration error histogram evaluation).

    Returns
    transform_matrix, kpts, error_abs_mean, iteration
    '''
    drmax = kwargs.get('drmax', 2)
    max_iter = kwargs.get('max_iter', 1000)
    remove_per_iter = kwargs.get('remove_per_iter', 1)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    start = kwargs.get('start', 'edges')
    estimation = kwargs.get('estimation', 'interval')

    transform_matrix = np.eye(3,3)
    iteration = 1
    max_error = drmax * 2.0
    errors = []
    while iteration <= max_iter and max_error > drmax:
        # determine the new transformation matrix     
        if TransformType == ShiftTransform:
            transform_matrix[0:2, 2] = np.mean(np.array(dst_pts.astype(float) - src_pts.astype(float)), axis=0)
            
        if TransformType == XScaleShiftTransform:
            n, d = src_pts.shape
            xsrc = np.array(src_pts)[:,0].astype(float)
            ysrc = np.array(src_pts)[:,1].astype(float)
            xdst = np.array(dst_pts)[:,0].astype(float)
            ydst = np.array(dst_pts)[:,1].astype(float)
            s00 = np.sum(xsrc)
            s01 = np.sum(xdst)
            sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
            #sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xdst*xdst) - s01*s01)
            #s10 = np.sum(ysrc)
            #s11 = np.sum(ydst)
            #sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)
            sy = 1.00 # force sy=1 if there are not enough keypoints points
            # spread over wide range of y-range (and y-range is small) to determine y-scale accuartely
            tx = np.mean(xdst) - sx * np.mean(xsrc)
            ty = np.mean(ydst) - sy * np.mean(ysrc)
            transform_matrix = np.array([[sx, 0,  tx],
                                         [0,  sy, ty],
                                         [0,  0,  1]])
            
        if TransformType == ScaleShiftTransform:
            n, d = src_pts.shape
            xsrc = np.array(src_pts)[:,0].astype(float)
            ysrc = np.array(src_pts)[:,1].astype(float)
            xdst = np.array(dst_pts)[:,0].astype(float)
            ydst = np.array(dst_pts)[:,1].astype(float)
            s00 = np.sum(xsrc)
            s01 = np.sum(xdst)
            sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
            s10 = np.sum(ysrc)
            s11 = np.sum(ydst)
            sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)
            tx = np.mean(xdst) - sx * np.mean(xsrc)
            ty = np.mean(ydst) - sy * np.mean(ysrc)
            transform_matrix = np.array([[sx, 0,  tx],
                                         [0,  sy, ty],
                                         [0,  0,  1]])
        
        if TransformType == AffineTransform:
            #estimate_scale = True
            #transform_matrix = _umeyama(src_pts, dst_pts, estimate_scale)
            tform = AffineTransform()
            tform.estimate(src_pts, dst_pts)
            transform_matrix = tform.params

        if TransformType == RegularizedAffineTransform:
            tform = RegularizedAffineTransform()
            tform.estimate(src_pts, dst_pts)  # regularization parameters are already part of estimate procedure 
            # this is implemented this way because the other code - RANSAC does not work otherwise
            transform_matrix = tform.params
        
        # estimate transformation errors and find outliers
        errs, xshifts, yshifts = estimate_kpts_transform_error(src_pts, dst_pts, transform_matrix)
        for j in np.arange(remove_per_iter):
            max_error = np.max(errs)
            ind = np.argmax(errs)
            src_pts = np.delete(src_pts, ind, axis=0)
            dst_pts = np.delete(dst_pts, ind, axis=0)
            errs = np.delete(errs, ind)
            #print('Iteration {:d}, max_error={:.2f} '.format(iteration, max_error), (iteration <= max_iter), (max_error > drmax))
        iteration +=1
    kpts = [src_pts, dst_pts]
    error_abs_mean = np.mean(np.abs(np.delete(errs, ind, axis=0)))
    xcounts, xbins = np.histogram(xshifts, bins=64)
    error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, estimation=estimation, start=start)
    ycounts, ybins = np.histogram(yshifts, bins=64)
    error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, estimation=estimation, start=start)
    return transform_matrix, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration


def determine_transformations_files(params_dsf):
    ''' Determine the transformation matrix from two sets of Key-Points and Descriptors.
    ©G.Shtengel, 09/2021. gleb.shtengel@gmail.com

    This is a faster version of the procedure - it loads the keypoints and matches for each frame from files.
    params_dsf = fnm_1, fnm_2, kwargs
    where 
    fnm_1 - keypoints for the first image (source)
    fnm_2 - keypoints for the first image (destination)
    and kwargs must include:
    use_existing_data : boolean
        Default is False. If True and this had already been performed, use existing results.
    TransformType - transformation type to be used (ShiftTransform, XScaleShiftTransform, ScaleShiftTransform, AffineTransform, RegularizedAffineTransform)
    BF_Matcher -  if True - use BF matcher, otherwise use FLANN matcher for keypoint matching
    solver - a string indicating which solver to use:
    'LinReg' will use Linear Regression with iterative "Throwing out the Worst Residual" Heuristic
    'RANSAC' will use RANSAC (Random Sample Consensus) algorithm.
    Lowe_Ratio_Threshold - threshold for Lowe's Ratio Test
    drmax - in the case of 'LinReg' - outlier threshold for iterative regression
           - in the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
    max_iter - max number of iterations
    save_matches - if True - save the matched keypoints into a binary dump file
    start : string
        'edges' (default) or 'center'. Start of search (registration error histogram evaluation).
    estimation : string
        'interval' (default) or 'count'. Returns a width of interval determined using search direction from above or total number of bins above half max (registration error histogram evaluation).

    Returns:
    transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration
    '''
    fnm_1, fnm_2, kwargs = params_dsf

    use_existing_data = kwargs.get('use_existing_data', False)
    ftype = kwargs.get("ftype", 0)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    kwargs['drmax'] = drmax
    max_iter = kwargs.get("max_iter", 1000)
    kwargs['max_iter'] = max_iter
    remove_per_iter = kwargs.get('remove_per_iter', 1)
    kwargs['remove_per_iter'] = remove_per_iter
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)    # threshold for Lowe's Ratio Test
    RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.005)  # fraction of data points for initial RANSAC iteration step.
    start = kwargs.get('start', 'edges')
    estimation = kwargs.get('estimation', 'interval')

    if save_matches:
        fnm_matches = fnm_2.replace('_kpdes.bin', '_matches.bin')
    else:
        fnm_matches = ''

    try_existing = use_existing_data and os.path.exists(fnm_matches)

    if try_existing:
        try:
            transform_matrix, fnm_matches_loc, kpts, error_abs_mean_loc, error_FWHMx_loc, error_FWHMy_loc, iteration_loc = pickle.load(open(fnm_matches, 'rb'))
            error_abs_mean = error_abs_mean_loc
            error_FWHMx = error_FWHMx_loc
            error_FWHMy = error_FWHMy_loc
            iteration = iteration_loc
        except:
            try_existing = False

    if not try_existing:
        if TransformType == RegularizedAffineTransform:
            def estimate(self, src, dst):
                self.params = determine_regularized_affine_transform(src, dst, l2_matrix, targ_vector)
            RegularizedAffineTransform.estimate = estimate

        kpp1s, des1 = pickle.load(open(fnm_1, 'rb'))
        kpp2s, des2 = pickle.load(open(fnm_2, 'rb'))
        
        kp1 = [list_to_kp(kpp1) for kpp1 in kpp1s]     # this converts a list of lists to a list of keypoint objects to be used by a matcher later
        kp2 = [list_to_kp(kpp2) for kpp2 in kpp2s]     # same for the second frame
        
        # establish matches
        if BFMatcher:    # if BFMatcher==True - use BF (Brute Force) matcher
            # This procedure uses BF (Brute-Force) Matcher.
            # BF matcher takes the descriptor of one feature in the first image and matches it with all other features
            # in second image using some distance calculation. The closest match in teh second image is returned.
            # For BF matcher, first we have to create the cv.DescriptorMatcher object with BFMatcher as type.
            # It takes two optional params:
            #
            # First parameter one is NormType. It specifies the distance measurement to be used. By default, it is L2.
            # It is good for SIFT, SURF, etc. (L1 is also there).
            # For binary string-based descriptors like ORB, BRIEF, BRISK, etc., Hamming should be used,
            # which uses Hamming distance as measurement. If ORB is using WTA_K of 3 or 4, Hamming2 should be used.
            #
            # Second parameter is boolean variable, CrossCheck which is false by default.
            # If it is true, Matcher returns only those matches with value (i,j)
            # such that i-th descriptor in set A has j-th descriptor in set B as the best match and vice-versa.
            # That is, the two features in both sets should match each other.
            # It provides consistant result, and is a good alternative to ratio test proposed by D.Lowe in SIFT paper.
            # http://amroamroamro.github.io/mexopencv/opencv_contrib/SURF_descriptor.html
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2,k=2)
        else:            # otherwise - use FLANN matcher
            # This procedure uses FLANN (Fast Library for Approximate Nearest Neighbors) Matcher (FlannBasedMatcher):
            # https://docs.opencv.org/3.4/d5/d6f/tutorial_feature_flann_matcher.html
            #
            # FLANN contains a collection of algorithms optimized for fast nearest neighbor search in large datasets
            # and for high dimensional features. It works faster than BFMatcher for large datasets.
            # For FlannBasedMatcher, it accepts two sets of options which specifies the algorithm to be used, its related parameters etc.
            # First one is Index. For various algorithms, the information to be passed is explained in FLANN docs.
            # http://amroamroamro.github.io/mexopencv/opencv_contrib/SURF_descriptor.html
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's Ratio test
        good = []
        for m, n in matches:
            if m.distance < Lowe_Ratio_Threshold * n.distance:
                good.append(m)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)
        
        if solver == 'LinReg':
            # Determine the transformation matrix via iterative liear regression
            transform_matrix, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration = determine_transformation_matrix(src_pts, dst_pts, **kwargs)
            n_kpts = len(kpts[0])
        else:  # the other option is solver = 'RANSAC'
            try:
                min_samples = np.int32(len(src_pts)*RANSAC_initial_fraction)
                model, inliers = ransac((src_pts, dst_pts),
                    TransformType, min_samples = min_samples,
                    residual_threshold = drmax, max_trials = max_iter)
                n_inliers = np.sum(inliers)
                inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
                inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
                placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
                src_pts_ransac = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
                dst_pts_ransac = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
                #non_nan_inds = ~np.isnan(src_pts_ransac) * ~np.isnan(dst_pts_ransac)
                #src_pts_ransac = src_pts_ransac[non_nan_inds]
                #dst_pts_ransac = dst_pts_ransac[non_nan_inds]
                kpts = [src_pts_ransac, dst_pts_ransac]
                # find shift parameters
                transform_matrix = model.params
                iteration = len(src_pts)- len(src_pts_ransac)
                reg_errors, xshifts, yshifts = estimate_kpts_transform_error(src_pts_ransac, dst_pts_ransac, transform_matrix)
                error_abs_mean = np.mean(np.abs(reg_errors))
                xcounts, xbins = np.histogram(xshifts, bins=64)
                error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, estimation=estimation, start=start)
                ycounts, ybins = np.histogram(yshifts, bins=64)
                error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, estimation=estimation, start=start)
            except:
                transform_matrix = np.eye(3)
                kpts = [[], []]
                error_abs_mean = np.nan
                iteration = 0
                error_FWHMx = np.nan
                error_FWHMy = np.nan
        if save_matches:
            int_results = [transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration]
            pickle.dump(int_results, open(fnm_matches, 'wb'))

    return transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration


def build_filename(fname, **kwargs):
    ftype = kwargs.get("ftype", 0)
    dtp = kwargs.get("dtp", np.int16)                             #  int16 or uint8
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    save_res_png  = kwargs.get("save_res_png", True)
    zbin_factor =  kwargs.get("zbin_factor", 1)             # binning factor in z-direction (milling direction). Default is 1
    preserve_scales =  kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
    subtract_FOVtrend_from_fit = kwargs.get("subtract_FOVtrend_from_fit", [True, True]) 

    pad_edges =  kwargs.get("pad_edges", True)
    suffix =  kwargs.get("suffix", '')

    frame = FIBSEM_frame(fname, ftype=ftype, calculate_scaled_images=False)
    dformat_read = 'I8' if frame.EightBit else 'I16'

    if dtp == np.int16:
        dformat_save = 'I16'
        fnm_reg = 'Registered_I16.mrc'
    else:
        dformat_save = 'I8'
        fnm_reg = 'Registered_I8.mrc'

    if zbin_factor>1:
        fnm_reg = fnm_reg.replace('.mrc', '_zbin{:d}.mrc'.format(zbin_factor))
            
    fnm_reg = fnm_reg.replace('.mrc', ('_' + TransformType.__name__ + '_' + solver + '.mrc'))

    fnm_reg = fnm_reg.replace('.mrc', '_drmax{:.1f}.mrc'.format(drmax))
  
    if preserve_scales:
        try:
            fnm_reg = fnm_reg.replace('.mrc', '_const_scls_'+fit_params[0]+'.mrc')
        except:
            pass

    if np.any(subtract_linear_fit):
        fnm_reg = fnm_reg.replace('.mrc', '_shift_subtr.mrc')
    
    if pad_edges:
        fnm_reg = fnm_reg.replace('.mrc', '_padded.mrc')

    if len(suffix)>0:
        fnm_reg = fnm_reg.replace('.mrc', '_' + suffix + '.mrc')
    return fnm_reg, dtp


def find_fit(tr_matr_cum, **kwargs):
    fit_params = kwargs.get('fit_params', ['SG', 11, 3])
    verbose = kwargs.get('verbose', False)
    if verbose:
        print('Using Fit Parameters ', fit_params, ' for processing the transformation matrix')
    fit_method = fit_params[0]
    if fit_method == 'SG':  # perform Savitsky-Golay fitting with parameters
        ws, porder = fit_params[1:3]         # window size 701, polynomial order 3
        s00_fit = savgol_filter(tr_matr_cum[:, 0, 0].astype(np.double), ws, porder)
        s01_fit = savgol_filter(tr_matr_cum[:, 0, 1].astype(np.double), ws, porder)
        s10_fit = savgol_filter(tr_matr_cum[:, 1, 0].astype(np.double), ws, porder)
        s11_fit = savgol_filter(tr_matr_cum[:, 1, 1].astype(np.double), ws, porder)
    else:
        fr = np.arange(0, len(tr_matr_cum), dtype=np.double)
        if fit_method == 'PF':  # perform polynomial fitting with parameters
            porder = fit_params[1]         # polynomial order
            s00_coeffs = np.polyfit(fr, tr_matr_cum[:, 0, 0].astype(np.double), porder)
            s00_fit = np.polyval(s00_coeffs, fr)
            s01_coeffs = np.polyfit(fr, tr_matr_cum[:, 0, 1].astype(np.double), porder)
            s01_fit = np.polyval(s01_coeffs, fr)
            s10_coeffs = np.polyfit(fr, tr_matr_cum[:, 1, 0].astype(np.double), porder)
            s10_fit = np.polyval(s10_coeffs, fr)
            s11_coeffs = np.polyfit(fr, tr_matr_cum[:, 1, 1].astype(np.double), porder)
            s11_fit = np.polyval(s11_coeffs, fr)
        
        else:   # otherwise perform linear fit with origin point tied to 1 for Sxx and Syy and to 0 for Sxy and Syx
            slp00 = -1.0 * (np.sum(fr)-np.dot(tr_matr_cum[:, 0, 0],fr))/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s00_fit = 1.0 + slp00 * fr
            slp11 = -1.0 * (np.sum(fr)-np.dot(tr_matr_cum[:, 1, 1],fr))/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s11_fit = 1.0 + slp11 * fr
            slp01 = np.dot(tr_matr_cum[:, 0, 1],fr)/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s01_fit = slp01 * fr
            slp10 = np.dot(tr_matr_cum[:, 1, 0],fr)/np.dot(fr,fr) # find the slope of a linear fit with forced first scale=1
            s10_fit = slp10 * fr
        
    tr_matr_cum_new = tr_matr_cum.copy()
    tr_matr_cum_new[:, 0, 0] = tr_matr_cum[:, 0, 0].astype(np.double) + 1.0 - s00_fit
    tr_matr_cum_new[:, 0, 1] = tr_matr_cum[:, 0, 1].astype(np.double) - s01_fit
    tr_matr_cum_new[:, 1, 0] = tr_matr_cum[:, 1, 0].astype(np.double) - s10_fit
    tr_matr_cum_new[:, 1, 1] = tr_matr_cum[:, 1, 1].astype(np.double) + 1.0 - s11_fit
    s_fits = [s00_fit, s01_fit, s10_fit, s11_fit]
    return tr_matr_cum_new, s_fits


def process_transformation_matrix_dataset(transformation_matrix, FOVtrend_x, FOVtrend_y, fnms_matches, npts, error_abs_mean, error_FWHMx, error_FWHMy, **kwargs):
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    Sample_ID = kwargs.get("Sample_ID", '')

    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    save_res_png  = kwargs.get("save_res_png", True)
    verbose = kwargs.get('verbose', False)

    preserve_scales =  kwargs.get("preserve_scales", True)  # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
    fit_params =  kwargs.get("fit_params", False)           # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                            # window size 701, polynomial order 3
    subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # The linear slopes along X- and Y- directions (respectively) will be subtracted from the cumulative shifts.
    subtract_FOVtrend_from_fit = kwargs.get("subtract_FOVtrend_from_fit", [True, True]) 
    if verbose:
        print("preserve_scales:", preserve_scales)
        print("subtract_linear_fit:", subtract_linear_fit)
        print("subtract_FOVtrend_from_fit:", subtract_FOVtrend_from_fit)

    pad_edges =  kwargs.get("pad_edges", True)

    tr_matr_cum = transformation_matrix.copy()   
    prev_mt = np.eye(3,3)
    for j, cur_mt in enumerate(tqdm(transformation_matrix, desc='Calculating Cummilative Transformation Matrix')):
        if np.any(np.isnan(cur_mt)):
            print('Frame: {:d} has ill-defined transformation matrix, will use identity transformation instead:'.format(j))
            print(cur_mt)
        else:
            prev_mt = np.matmul(cur_mt, prev_mt)
        tr_matr_cum[j] = prev_mt
    # Now insert identity matrix for the zero frame which does not need to be trasformed

    tr_matr_cum = np.insert(tr_matr_cum, 0, np.eye(3,3), axis=0)  
    
    fr = np.arange(0, len(tr_matr_cum), dtype=np.double)
    s00_cum_orig = tr_matr_cum[:, 0, 0].astype(np.double)
    s01_cum_orig = tr_matr_cum[:, 0, 1].astype(np.double)
    s10_cum_orig = tr_matr_cum[:, 1, 0].astype(np.double)
    s11_cum_orig = tr_matr_cum[:, 1, 1].astype(np.double)
    Xshift_cum_orig = tr_matr_cum[:, 0, 2].astype(np.double)
    Yshift_cum_orig = tr_matr_cum[:, 1, 2].astype(np.double)
    columns_shifts=['s00_cum_orig', 's11_cum_orig', 's01_cum_orig', 's10_cum_orig', 'Xshift_cum_orig', 'Yshift_cum_orig']
    shifts_dt = pd.DataFrame(np.vstack((s00_cum_orig, s11_cum_orig, s01_cum_orig, s10_cum_orig, Xshift_cum_orig, Yshift_cum_orig)).T, columns = columns_shifts, index = None)
        
    if preserve_scales:  # in case of transformation WITH scale perservation
        if verbose:
            print('Recalculating the transformation matrix for preserved scales')
        tr_matr_cum, s_fits = find_fit(tr_matr_cum, fit_params=fit_params, verbose=verbose)
        s00_fit, s01_fit, s10_fit, s11_fit = s_fits
        columns_shifts_add=['s00_fit', 's11_fit', 's01_fit', 's10_fit']
        shifts_dt_add = pd.DataFrame(np.vstack((s00_fit, s11_fit, s01_fit, s10_fit)).T, columns = columns_shifts_add, index = None)
        shifts_dt = pd.concat([shifts_dt, shifts_dt_add], axis=1)
        txs = np.zeros(len(tr_matr_cum), dtype=float)
        tys = np.zeros(len(tr_matr_cum), dtype=float)
        
        failed_to_open_matches = 0
        failed_to_open_fnms = []
        for j, fnm_matches in enumerate(tqdm(fnms_matches, desc='Recalculating the shifts for preserved scales: ')):
            try:
                transform_matrix, fnm_matches_loc, [src_pts, dst_pts], error_abs_mean_loc, error_FWHMx_loc, error_FWHMy_loc, iteration_loc = pickle.load(open(fnm_matches, 'rb'))
                try:
                    txs[j+1] = np.mean(tr_matr_cum[j, 0, 0] * dst_pts[:, 0] + tr_matr_cum[j, 0, 1] * dst_pts[:, 1]
                                       - tr_matr_cum[j+1, 0, 0] * src_pts[:, 0] - tr_matr_cum[j+1, 0, 1] * src_pts[:, 1])
                    tys[j+1] = np.mean(tr_matr_cum[j, 1, 1] * dst_pts[:, 1] + tr_matr_cum[j, 1, 0] * dst_pts[:, 0]
                                       - tr_matr_cum[j+1, 1, 1] * src_pts[:, 1] - tr_matr_cum[j+1, 1, 0] * src_pts[:, 0])
                except:
                    pass
            except:
                failed_to_open_matches += 1
                failed_to_open_fnms.append(fnm_matches)
                
        if failed_to_open_matches > 0:
            if verbose:
                print('Failed to open {:d} files containgng matched keypoints'.format(failed_to_open_matches))
                failed_fnms_file = os.path.join(data_dir, 'Failed_to_read_fnms.txt')
                print('Saving the filenames of failed to open files into: ', failed_fnms_file)
                with open(failed_fnms_file, 'w') as f:
                    for line in failed_to_open_fnms:
                        f.write(f"{line}\n")
                print('Transformation Matrix optimization will most likely not work')
        txs_cum = np.cumsum(txs)
        tys_cum = np.cumsum(tys)
        tr_matr_cum[:, 0, 2] = txs_cum
        tr_matr_cum[:, 1, 2] = tys_cum

    Xshift_cum = tr_matr_cum[:, 0, 2].copy()
    Yshift_cum = tr_matr_cum[:, 1, 2].copy()

    # Subtract linear trends from offsets
    if subtract_linear_fit[0]:
        fr = np.arange(0, len(Xshift_cum))
        if subtract_FOVtrend_from_fit[0]:
            pX = np.polyfit(fr, Xshift_cum+FOVtrend_x, 1)
        else:
            pX = np.polyfit(fr, Xshift_cum, 1)
        Xfit = np.polyval(pX, fr)
        Xshift_residual = Xshift_cum - Xfit
        #Xshift_residual0 = -np.polyval(pX, 0.0)
        columns_shifts_add=['Xshift_cum', 'Xfit']
        shifts_dt_add = pd.DataFrame(np.vstack((Xshift_cum, Xfit)).T, columns = columns_shifts_add, index = None)
        shifts_dt = pd.concat([shifts_dt, shifts_dt_add], axis=1)
    else:
        Xshift_residual = Xshift_cum.copy()
        Xfit = np.zeros(len(Xshift_cum))

    if subtract_linear_fit[1]:
        fr = np.arange(0, len(Yshift_cum))
        if subtract_FOVtrend_from_fit[1]:
            pY = np.polyfit(fr, Yshift_cum+FOVtrend_y, 1)
        else:
            pY = np.polyfit(fr, Yshift_cum, 1)
        Yfit = np.polyval(pY, fr)
        Yshift_residual = Yshift_cum - Yfit
        #Yshift_residual0 = -np.polyval(pY, 0.0)
        columns_shifts_add=['Yshift_cum', 'Yfit']
        shifts_dt_add = pd.DataFrame(np.vstack((Yshift_cum, Yfit)).T, columns = columns_shifts_add, index = None)
        shifts_dt = pd.concat([shifts_dt, shifts_dt_add], axis=1)
    else:
        Yshift_residual = Yshift_cum.copy()
        Yfit = np.zeros(len(Yshift_cum))

    # define new cumulative transformation matrix where the offests may have linear slopes subtracted
    tr_matr_cum[:, 0, 2] = Xshift_residual-Xshift_residual[0]
    tr_matr_cum[:, 1, 2] = Yshift_residual-Yshift_residual[0]


    # save the data
    default_bin_file = os.path.join(data_dir, fnm_reg.replace('.mrc', '_transf_matrix.bin'))
    transf_matrix_bin_file = kwargs.get("dump_filename", default_bin_file)
    transf_matrix_xlsx_file = default_bin_file.replace('.bin', '.xlsx')

    xlsx_writer = pd.ExcelWriter(transf_matrix_xlsx_file, engine='xlsxwriter')
    columns=['T00 (Sxx)', 'T01 (Sxy)', 'T02 (Tx)',  
                 'T10 (Syx)', 'T11 (Syy)', 'T12 (Ty)', 
                 'T20 (0.0)', 'T21 (0.0)', 'T22 (1.0)']
    tr_mx_dt = pd.DataFrame(transformation_matrix.reshape((len(transformation_matrix), 9)), columns = columns, index = None)
    tr_mx_dt.to_excel(xlsx_writer, index=None, sheet_name='Orig. Transformation Matrix')

    tr_mx_cum_dt = pd.DataFrame(tr_matr_cum.reshape((len(tr_matr_cum), 9)), columns = columns, index = None)
    tr_mx_cum_dt.to_excel(xlsx_writer, index=None, sheet_name='Cum. Transformation Matrix')

    shifts_dt.to_excel(xlsx_writer, index=None, sheet_name='Intermediate Results')
    
    columns_reg_stat = ['Npts', 'Mean Abs Error', 'Xerror_FWHM', 'Yerror_FWHM']
    reg_stat_dt = pd.DataFrame(np.vstack((npts, error_abs_mean, np.array(error_FWHMx), np.array(error_FWHMy))).T, columns = columns_reg_stat, index = None)
    reg_stat_dt.to_excel(xlsx_writer, index=None, sheet_name='Reg. Stat. Info')

    kwargs_info = pd.DataFrame([kwargs]).T   # prepare to be save in transposed format
    kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')

    #xlsx_writer.save()
    xlsx_writer.close()

    '''
    DumpObject = [kwargs, npts, error_abs_mean,
              transformation_matrix, s00_cum_orig, s11_cum_orig, s00_fit, s11_fit,
              tr_matr_cum, s01_cum_orig, s10_cum_orig, s01_fit, s10_fit,
              Xshift_cum_orig, Yshift_cum_orig, Xshift_cum, Yshift_cum, Yshift_cum, Xfit, Yfit]
    with open(transf_matrix_bin_file,"wb") as f:
        pickle.dump(DumpObject, f)
    '''
    return tr_matr_cum, transf_matrix_xlsx_file


def calculate_residual_deformation_fields_dataset(tr_matr_cum, image_shape, fnms_matches, **kwargs):
    '''
    Calculates residual deformation fields for transformation IN ADDITION to that determined by transformation_matrix. ©G.Shtengel 01/2025 gleb.shtengel@gmail.com

    Parameters
    ----------
    tr_matr_cum : transformation matrix
    image_shape : target image shape
    fnms_matches : files with point matches

    kwargs:
    ----------
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
    zero_mean : boolean
        If True (Default), mean value of the deformation field is subtracted for each frame
    data_dir : str
        data directory (path)
    fnm_reg : str
        filename for the final registered dataset
    verbose : boolean

    Returns deformation_fields, deformation_fields_bin_file
    '''
    deformation_type = kwargs.get("deformation_type", 'post_1DY')
    deformation_sigma = kwargs.get('deformation_sigma', 50)
    zero_mean = kwargs.get('zero_mean', True)
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    verbose = kwargs.get('verbose', False)
    nfrs = len(fnms_matches)

    if verbose:
        print('calculate_residual_deformation_fields_dataset: will calulate residual deformation fields in format: ', deformation_type)
        print('calculate_residual_deformation_fields_dataset: deformation_sigma = ', deformation_sigma)
        print('calculate_residual_deformation_fields_dataset: zero_mean = ', zero_mean)

    if deformation_type == 'post_1DY':  # in case of transformation WITH scale perservation
        if verbose:
            print('Calculating the residual deformation fields for post_1DY deformation')
            print('The shape of the output array will be: ', nfrs, image_shape[0])
        '''
        There are nfrs matched pairs for nfrs+1 frames, and nfrs+1 transformation matrices.
        Even transformation_matrix[0] is NOT in general a unity matrix (because of scale preservation and shift subtraction compensations).
        We will need nfrs+1 deformation fields - same number as number of transformation matrices.
        But the deformation_fields[0] is a field of zeros, at least for now.
        '''
        deformation_fields = np.zeros((nfrs+1, image_shape[0]), dtype=float)
        for j, fnm_matches in enumerate(tqdm(fnms_matches, desc='Calculating the residual deformation fields for post_1DY deformation')):
            try:
                if verbose:
                    print('calculate_residual_deformation_fields_dataset: Step: ', j)
                    print(fnm_matches)
                    print(tr_matr_cum[j])
                transform_matrix, fnm_matches_loc, [src_pts, dst_pts], error_abs_mean_loc, error_FWHMx_loc, error_FWHMy_loc, iteration_loc = pickle.load(open(fnm_matches, 'rb'))
                deformation_fields[j+1] = determine_residual_deformation_field(src_pts, dst_pts, tr_matr_cum[j], tr_matr_cum[j+1], image_shape,
                                                                                deformation_type = '1DY',
                                                                                deformation_sigma = deformation_sigma,
                                                                                zero_mean=zero_mean,
                                                                                verbose=verbose)
            except:
                pass
        deformation_fields = np.cumsum(deformation_fields, axis = 0)  
    if verbose:
        print('calculate_residual_deformation_fields_dataset: output deformation_fields shape', deformation_fields.shape)
        print('calculate_residual_deformation_fields_dataset: average output deformation_field ', np.mean(deformation_fields))
    # save the data
    default_bin_file = os.path.join(data_dir, fnm_reg.replace('.mrc', '_deformation_fields.bin'))
    deformation_fields_bin_file = kwargs.get("dump_filename", default_bin_file)

    DumpObject = [kwargs, image_shape, deformation_fields]
    with open(deformation_fields_bin_file,"wb") as f:
        pickle.dump(DumpObject, f)
    
    return deformation_fields, deformation_fields_bin_file


def SIFT_find_keypoints_dataset(fr, **kwargs):
    '''
    Evaluate SIFT key point discovery for a test frame (fr). ©G.Shtengel 08/2022 gleb.shtengel@gmail.com
    
    Parameters:
    fr : str
        filename for the data frame to be used for SIFT key point discovery evaluation
    
    kwargs
    ---------
    data_dir : str
        data directory (path)
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    fnm_reg : str
        filename for the final registered dataset
    thr_min : float
        CDF threshold for determining the minimum data value
    thr_max : float
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for key-point extraction
        if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
    
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
        Note that the its meaning is different from the contrastThreshold,
        i.e. the larger the edgeThreshold, the less features are filtered out
        (more features are retained).
    SIFT_sigma : double
        SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
        If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check

    Returns:
    dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts
    '''

    ftype = kwargs.get("ftype", 0)
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    save_res_png  = kwargs.get("save_res_png", True)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.025)
    RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.005)  # fraction of data points for initial RANSAC iteration step.

    frame = FIBSEM_frame(fr, ftype=ftype, calculate_scaled_images=False)
    if ftype == 0:
        if frame.FileVersion > 8 :
            Sample_ID = frame.Sample_ID.strip('\x00')
        else:
            Sample_ID = frame.Notes[0:16]
    else:
        Sample_ID = frame.Sample_ID
    Sample_ID = kwargs.get("Sample_ID", Sample_ID)

    print(Sample_ID)
    
    #if save_res_png :
    #    frame.display_images()

    img = np.ravel(frame.RawImageA)
    fsz=12
    fszl=11
    dmin, dmax = frame.get_image_min_max(image_name = 'RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins)
    xi = dmin-(np.abs(dmax-dmin)/10)
    xa = dmax+(np.abs(dmax-dmin)/10)

    fig, axs = plt.subplots(2,1, figsize=(6,6))
    fig.suptitle(Sample_ID + ',  thr_min={:.0e}, thr_max={:.0e}, SIFT_contrastThreshold={:.3f}, comp.time={:.1f}sec'.format(thr_min, thr_max, SIFT_contrastThreshold, comp_time), fontsize=fszl)
    
    hist, bins, patches = axs[0].hist(img, bins = nbins)
    axs[0].set_xlim(xi, xa)
    axs[0].plot([dmin, dmin], [0, np.max(hist)], 'r', linestyle = '--')
    axs[0].plot([dmax, dmax], [0, np.max(hist)], 'g', linestyle = '--')
    axs[0].set_ylabel('Count', fontsize = fsz)
    pdf = hist / (frame.XResolution * frame.YResolution)
    cdf = np.cumsum(pdf)
    xCDF = bins[0:-1]+(bins[1]-bins[0])/2.0
    xthr = [xCDF[0], xCDF[-1]]
    ythr_min = [thr_min, thr_min]
    y1thr_max = [1-thr_max, 1-thr_max]

    axs[1].plot(xCDF, cdf, label='CDF')
    axs[1].plot(xthr, ythr_min, 'r', label='thr_min={:.5f}'.format(thr_min))
    axs[1].plot([dmin, dmin], [0, 1], 'r', linestyle = '--', label = 'data_min={:.1f}'.format(dmin))
    axs[1].plot(xthr, y1thr_max, 'g', label='1.0 - thr_max = {:.5f}'.format(1-thr_max))
    axs[1].plot([dmax, dmax], [0, 1], 'g', linestyle = '--', label = 'data_max={:.1f}'.format(dmax))
    axs[1].set_xlabel('Intensity Level', fontsize = fsz)
    axs[1].set_ylabel('CDF', fontsize = fsz)
    axs[1].set_xlim(xi, xa)
    axs[1].legend(loc='center', fontsize=fsz)
    axs[0].set_title('Data Min and Max with thr_min={:.0e},  thr_max={:.0e}'.format(thr_min, thr_max), fontsize = fsz)
    for ax in axs.ravel():
        ax.grid(True)
        
    t0 = time.time()
    params1 = [fr, dmin, dmax, kwargs]
    fnm_1 = extract_keypoints_descr_files(params1)
           
    t1 = time.time()
    comp_time = (t1-t0)
    #print('Time to compute: {:.1f}sec'.format(comp_time))
       
    xfsz = 3 * (int(7 * frame.XResolution / np.max([frame.XResolution, frame.YResolution]))+1)
    yfsz = 3 * (int(7 * frame.YResolution / np.max([frame.XResolution, frame.YResolution]))+2)
    fig2, ax = plt.subplots(1,1, figsize=(xfsz,yfsz))
    fig2.subplots_adjust(left=0.0, bottom=0.25*(1-frame.YResolution/frame.XResolution), right=1.0, top=1.0)
    symsize = 2
    fsize = 12  
    img2 = FIBSEM_frame(fr, ftype=ftype, calculate_scaled_images=False).RawImageA
    ax.imshow(img2, cmap='Greys', vmin=dmin, vmax=dmax)
    ax.axis(False)
    
    kpp1s, des1 = pickle.load(open(fnm_1, 'rb'))
    kp1 = [list_to_kp(kpp1) for kpp1 in kpp1s]     # this converts a list of lists to a list of keypoint objects to be used by a matcher later
    src_pts = np.float32([ kp.pt for kp in kp1 ]).reshape(-1, 2)    
    x, y = src_pts.T
    print('Extracted {:d} keyponts'.format(len(kp1)))
    # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
    vec_field = ax.scatter(x,y, s=0.02, marker='o', c='r')
    ax.text(0.01, 1.1-0.13*frame.YResolution/frame.XResolution, Sample_ID + ', thr_min={:.0e}, thr_max={:.0e}, SIFT_nfeatures={:d}'.format(thr_min, thr_max, SIFT_nfeatures), fontsize=fsize, transform=ax.transAxes)
    if save_res_png :
        png_name = os.path.splitext(fr)[0] + '_SIFT_kpts_eval_'+'_thr_min{:.5f}_thr_max{:.5f}.png'.format(thr_min, thr_max) 
        fig2.savefig(png_name, dpi=300)
    return(dmin, dmax, comp_time, src_pts)


# This is a function used for selecting proper SIFT and other parameters for processing
def SIFT_evaluation_dataset(fs, **kwargs):
    '''
    Evaluate SIFT settings and perfromance of few test frames (fs). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    
    Parameters:
    ---------
    fs : array of str
        filenames for the data frames to be used for SIFT evaluation
    
    kwargs
    ---------
    DASK_client : DASK client. If set to empty string '' (default), local computations are performed
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    use_DASK : boolean
    number_of_repeats : int
            number of repeats of the calculations (under the same conditions). Default is 1.
    data_dir : str
        data directory (path)
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    fnm_reg : str
        filename for the final registered dataset
    thr_min : float
        CDF threshold for determining the minimum data value
    thr_max : float
        CDF threshold for determining the maximum data value
    nbins : int
        number of histogram bins for building the PDF and CDF
    evaluation_box : list of 4 int
        evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
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
    targ_vector = 1D float array
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
        Note that the its meaning is different from the contrastThreshold,
        i.e. the larger the edgeThreshold, the less features are filtered out
        (more features are retained).
    SIFT_sigma : double
        SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
        If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check
    start : string
        'edges' (default) or 'center'. start of search.
    estimation : string
        'interval' (default) or 'count'. Returns a width of interval determined using search direction from above or total number of bins above half max
    memory_profiling : boolean
        If True will perfrom memory profiling. Default is False.
    use_existing_data : boolean
        Default is False. If True and this had already been performed, use existing results
    Returns:
    dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts, error_FWHMx, error_FWHMy
    '''
    memory_profiling = kwargs.get('memory_profiling', False)
    if memory_profiling:
        rss_before, vms_before, shared_before = get_process_memory()
        start_time = time.time()
    DASK_client = kwargs.get('DASK_client', '')
    if DASK_client == '':
        use_DASK = False
    else:
        use_DASK = kwargs.get('use_DASK', False)
    DASK_client_retries = kwargs.get("DASK_client_retries", 3)
    number_of_repeats = kwargs.get('number_of_repeats', 1)
    ftype = kwargs.get("ftype", 0)
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get("nbins", 256)
    evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.005)  # fraction of data points for initial RANSAC iteration step.
    drmax = kwargs.get("drmax", 2.0)
    max_iter = kwargs.get("max_iter", 1000)
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
    BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
    save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
    save_res_png  = kwargs.get("save_res_png", True)
    verbose = kwargs.get('verbose', False)
    SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = kwargs.get('SIFT_nOctaveLayers', 3)
    SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 0.025)
    SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 10.0)
    SIFT_sigma = kwargs.get('SIFT_sigma', 1.6)
    start = kwargs.get('start', 'edges')
    estimation = kwargs.get('estimation', 'interval')
    use_existing_data = kwargs.get('use_existing_data', False)

    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: Start of Execution: RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
    
    frame = FIBSEM_frame(fs[0], ftype=ftype, calculate_scaled_images=False)
    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: Read First Frame  : RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
    if ftype == 0:
        if frame.FileVersion > 8 :
            Sample_ID = frame.Sample_ID.strip('\x00')
        else:
            Sample_ID = frame.Notes[0:16]
    else:
        Sample_ID = frame.Sample_ID
    Sample_ID = kwargs.get("Sample_ID", Sample_ID)

    print("Sample ID:   ", Sample_ID)

    if BFMatcher:
        matcher = 'BFMatcher'
    else:
        matcher = 'FLANN'
    
    #if save_res_png :
    #    frame.display_images()

    img = np.ravel(frame.RawImageA)
    fsz = 12
    fszl = 11
    dmin, dmax = frame.get_image_min_max(image_name = 'RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins)
    xi = dmin-(np.abs(dmax-dmin)/10)
    xa = dmax+(np.abs(dmax-dmin)/10)
    
    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: Calculated Min/Max: RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))

    fig, axs = plt.subplots(2,2, figsize=(12,8))
    fig.suptitle(Sample_ID + ',  thr_min={:.0e}, thr_max={:.0e}, SIFT_contrastThreshold={:.3f}'.format(thr_min, thr_max, SIFT_contrastThreshold), fontsize=fszl)

    hist, bins, patches = axs[0,0].hist(img, bins = nbins)
    axs[0,0].set_xlim(xi, xa)
    axs[0,0].plot([dmin, dmin], [0, np.max(hist)], 'r', linestyle = '--')
    axs[0,0].plot([dmax, dmax], [0, np.max(hist)], 'g', linestyle = '--')
    axs[0,0].set_ylabel('Count', fontsize = fsz)
    pdf = hist / (frame.XResolution * frame.YResolution)
    cdf = np.cumsum(pdf)
    xCDF = bins[0:-1]+(bins[1]-bins[0])/2.0
    xthr = [xCDF[0], xCDF[-1]]
    ythr_min = [thr_min, thr_min]
    y1thr_max = [1-thr_max, 1-thr_max]

    axs[1,0].plot(xCDF, cdf, label='CDF')
    axs[1,0].plot(xthr, ythr_min, 'r', label='thr_min={:.5f}'.format(thr_min))
    axs[1,0].plot([dmin, dmin], [0, 1], 'r', linestyle = '--', label = 'data_min={:.1f}'.format(dmin))
    axs[1,0].plot(xthr, y1thr_max, 'g', label='1.0 - thr_max = {:.5f}'.format(1-thr_max))
    axs[1,0].plot([dmax, dmax], [0, 1], 'g', linestyle = '--', label = 'data_max={:.1f}'.format(dmax))
    axs[1,0].set_xlabel('Intensity Level', fontsize = fsz)
    axs[1,0].set_ylabel('CDF', fontsize = fsz)
    axs[1,0].set_xlim(xi, xa)
    axs[1,0].legend(loc='center', fontsize=fsz)
    axs[0,0].set_title('Data Min and Max with thr_min={:.0e},  thr_max={:.0e}'.format(thr_min, thr_max), fontsize = fsz)

    minmax = []
    for j,f in enumerate(fs):
        minmax.append(FIBSEM_frame(f, ftype=ftype, calculate_scaled_images=False).get_image_min_max(image_name = 'RawImageA', thr_min=thr_min, thr_max=thr_max, nbins=nbins))
        if memory_profiling:
            elapsed_time = elapsed_since(start_time)
            rss_after, vms_after, shared_after = get_process_memory()
            print("Profiling: Re-calc {:d} Min/Max : RSS: {:>8} | VMS: {:>8} | SHR {"
                  ":>8} | time: {:>8}"
                .format(j, format_bytes(rss_after - rss_before),
                        format_bytes(vms_after - vms_before),
                        format_bytes(shared_after - shared_before),
                        elapsed_time))
    dmin = np.min(np.array(minmax))
    dmax = np.max(np.array(minmax))
    #print('data range: ', dmin, dmax)
    t0 = time.time()

    params1 = [fs[0], dmin, dmax, kwargs]
    fnm_1 = extract_keypoints_descr_files(params1)
    kpp1s, des1 = pickle.load(open(fnm_1, 'rb'))
    n_kpts = len(kpp1s)
    params2 = [fs[1], dmin, dmax, kwargs]
    fnm_2 = extract_keypoints_descr_files(params2)

    kwargs.pop('DASK_client', None)
    params_dsf = [fnm_1, fnm_2, kwargs]
    params_dsf_mult = []
    
    for j in np.arange(number_of_repeats):
        kwargs_temp = kwargs.copy()
        kwargs_temp['iteration'] = j
        params_dsf_mult.append([fnm_1, fnm_2, kwargs_temp])

    n_matches_tot = []
    if use_DASK:
        futures = DASK_client.map(determine_transformations_files, params_dsf_mult)
        results = DASK_client.gather(futures)
    else:
        results = []
        for j in tqdm(np.arange(number_of_repeats), desc='Repeating SIFT calculation {:d} times'.format(number_of_repeats)):
            results.append(determine_transformations_files(params_dsf))
            if memory_profiling:
                elapsed_time = elapsed_since(start_time)
                rss_after, vms_after, shared_after = get_process_memory()
                print("Profiling: Extr.kpts try {:d}: RSS: {:>8} | VMS: {:>8} | SHR {"
                      ":>8} | time: {:>8}"
                    .format(j, format_bytes(rss_after - rss_before),
                            format_bytes(vms_after - vms_before),
                            format_bytes(shared_after - shared_before),
                            elapsed_time))

    n_matches_tot = np.array([len(res[2][0]) for res in results])
    transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration = results[np.argmin(n_matches_tot)]
    n_matches = len(kpts[0])
    print('')
    if number_of_repeats > 1:
        print('Repeated registration calculations {:d} times'.format(number_of_repeats))
        print('Average # of detected matches: {:.2f}'.format(np.mean(n_matches_tot)))
        print('min # of detected matches: {:d}'.format(np.min(n_matches_tot)))
        print('STD # of detected matches: {:.2f}'.format(np.std(n_matches_tot)))
    else:
        print('# of detected matches: {:d}'.format(n_matches))
    print('')
    #print('Transformation Matrix: ')
    #print(transform_matrix)
    if n_matches > 0:
        src_pts_filtered, dst_pts_filtered = kpts
        src_pts_transformed = src_pts_filtered @ transform_matrix[0:2, 0:2].T + transform_matrix[0:2, 2]
        xshifts = (dst_pts_filtered - src_pts_transformed)[:,0]
        yshifts = (dst_pts_filtered - src_pts_transformed)[:,1]
    
    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: Finished Calcs.   : RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
    t1 = time.time()
    comp_time = (t1-t0)
    #print('Time to compute: {:.1f}sec'.format(comp_time))

    axx = axs[0,1]
    axx.set_xlabel('SIFT: X Error (pixels)')
    axy = axs[1,1]
    axy.set_xlabel('SIFT: Y Error (pixels)')
    if n_matches > 1:
        xcounts, xbins, xhist_patches = axx.hist(xshifts, bins=64)
        error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, estimation=estimation, start=start, max_aver_aperture=5)
        dbx = (xbins[1]-xbins[0])/2.0
        #axx.plot([xbins[indxi]+dbx, xbins[indxa]+dbx], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
        axx.plot([xbins[indxi], xbins[indxa]], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
        axx.plot([xbins[mxx_ind]+dbx], [mxx], 'rd')
        axx.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(xshifts)), transform=axx.transAxes, fontsize=fsz)
        axx.text(0.05, 0.8, 'median={:.3f}'.format(np.median(xshifts)), transform=axx.transAxes, fontsize=fsz)
        axx.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMx), transform=axx.transAxes, fontsize=fsz)
        axx.set_title('data range: {:.1f} ÷ {:.1f}'.format(dmin, dmax), fontsize=fsz)
        ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64)
        error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, estimation=estimation, start=start, max_aver_aperture=5)
        dby = (ybins[1]-ybins[0
            ])/2.0
        #axy.plot([ybins[indyi] + dby, ybins[indya] + dby], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
        axy.plot([ybins[indyi], ybins[indya]], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
        axy.plot([ybins[mxy_ind] + dby], [mxy], 'rd')
        axy.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(yshifts)), transform=axy.transAxes, fontsize=fsz)
        axy.text(0.05, 0.8, 'median={:.3f}'.format(np.median(yshifts)), transform=axy.transAxes, fontsize=fsz)
        axy.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMy), transform=axy.transAxes, fontsize=fsz)
    else:
        axx.text(0.05, 0.9, '{:d} Matches Detected'.format(n_matches), transform=axx.transAxes, fontsize=fsz)
        axy.text(0.05, 0.9, '{:d} Matches Detected'.format(n_matches), transform=axy.transAxes, fontsize=fsz)

    axt=axx  # print Transformation Matrix data over axx plot
    axt.text(0.65, 0.8, 'Transf. Matrix:', transform=axt.transAxes, fontsize=fsz)
    axt.text(0.55, 0.7, '{:.4f} {:.4f} {:.4f}'.format(transform_matrix[0,0], transform_matrix[0,1], transform_matrix[0,2]), transform=axt.transAxes, fontsize=fsz-1)
    axt.text(0.55, 0.6, '{:.4f} {:.4f} {:.4f}'.format(transform_matrix[1,0], transform_matrix[1,1], transform_matrix[1,2]), transform=axt.transAxes, fontsize=fsz-1)
    
    for ax in axs.ravel():
        ax.grid(True)
    
    fig.suptitle(Sample_ID + ',  thr_min={:.0e}, thr_max={:.0e}, comp.time={:.1f}sec'.format(thr_min, thr_max, comp_time), fontsize=fszl)

    if TransformType == RegularizedAffineTransform:
        tstr = ['{:d}'.format(x) for x in targ_vector] 
        otext =  TransformType.__name__ + ', λ= {:.1e}, t=['.format(l2_matrix[0,0]) + ', '.join(tstr) + '], ' + solver + ', #of matches={:d}'.format(n_matches)
    else:
        otext = TransformType.__name__ + ', ' + solver + ', #of matches={:d}'.format(n_matches)

    axs[0,0].text(0.01, 1.14, otext, fontsize=fszl, transform=axs[0,0].transAxes)        
    if save_res_png :
        png_name = os.path.join(data_dir, (os.path.splitext(os.path.split(fs[0])[-1])[0] + '_SIFT_eval_'+TransformType.__name__ + '_' + solver +'_thr_min{:.0e}_thr_max{:.0e}.png'.format(thr_min, thr_max)))
        fig.savefig(png_name, dpi=300)
    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: Plotted Histograms: RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))

    xfsz = int(7 * frame.XResolution / np.max([frame.XResolution, frame.YResolution]))+1
    yfsz = int(7 * frame.YResolution / np.max([frame.XResolution, frame.YResolution]))+2
    fig2, ax = plt.subplots(1,1, figsize=(xfsz,yfsz))
    fig2.subplots_adjust(left=0.0, bottom=0.25*(1-frame.YResolution/frame.XResolution), right=1.0, top=1.0)
    symsize = 2
    fsize_text = 6
    fsize_label = 10
    #img2 = FIBSEM_frame(fs[-1], ftype=ftype, calculate_scaled_images=False).RawImageA
    #ax.imshow(img2, cmap='Greys', vmin=dmin, vmax=dmax)
    ax.imshow(frame.RawImageA, cmap='Greys', vmin=dmin, vmax=dmax)
    ax.axis(False)
    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: Plotted Raw Image : RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
    
    if n_matches > 0:
        x, y = dst_pts_filtered.T
        M = np.sqrt(xshifts*xshifts+yshifts*yshifts)
        xs = xshifts
        ys = yshifts
        # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
        vec_field = ax.quiver(x,y,xs,ys,M, scale=50, width =0.0015, cmap='jet')
        cbar = fig2.colorbar(vec_field, pad=0.05, shrink=0.70, orientation = 'horizontal', format="%.1f")
        cbar.set_label('SIFT Shift Amplitude (pix)', fontsize=fsize_label)

    ax.text(0.005, 1.00 - 0.010*frame.XResolution/frame.YResolution, fs[0], fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.023*frame.XResolution/frame.YResolution, Sample_ID, fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.036*frame.XResolution/frame.YResolution, 'thr_min={:.0e}, thr_max={:.0e}'.format(thr_min, thr_max), fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.049*frame.XResolution/frame.YResolution, TransformType.__name__+ ', ' + solver + ',  ' + matcher, fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.062*frame.XResolution/frame.YResolution, 'SIFT_nfeatures={:d}'.format(SIFT_nfeatures), fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.075*frame.XResolution/frame.YResolution, 'SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold), fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.088*frame.XResolution/frame.YResolution, 'SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma), fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.101*frame.XResolution/frame.YResolution, 'RANSAC_initial_fraction={:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter), fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.114*frame.XResolution/frame.YResolution, 'drmax={:.3f}'.format(drmax), fontsize=fsize_text, transform=ax.transAxes)
    ax.text(0.005, 1.00 - 0.127*frame.XResolution/frame.YResolution, '# of keypoints = {:d}, # of matches ={:d}'.format(n_kpts, n_matches), fontsize=fsize_text, transform=ax.transAxes)
    if verbose:
        print('thr_min={:.0e}, thr_max={:.0e}'.format(thr_min, thr_max))
        print(TransformType.__name__+ ', ' + solver + ',  ' + matcher)
        print('SIFT_nfeatures={:d}'.format(SIFT_nfeatures))
        print('SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold))
        print('SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma))
        print('RANSAC_initial_fraction = {:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter))
        print('drmax={:.3f}'.format(drmax))
        print('# of keypoints = {:d}, # of matches ={:d}'.format(n_kpts, n_matches))

    if save_res_png :
        fig2_fnm = os.path.join(data_dir, (os.path.splitext(os.path.split(fs[0])[-1])[0]+'_SIFT_vmap_'+TransformType.__name__ + '_' + solver +'_thr_min{:.0e}_thr_max{:.0e}.png'.format(thr_min, thr_max)))
        fig2.savefig(fig2_fnm, dpi=600)
    if memory_profiling:
        elapsed_time = elapsed_since(start_time)
        rss_after, vms_after, shared_after = get_process_memory()
        print("Profiling: End of Execution  : RSS: {:>8} | VMS: {:>8} | SHR {"
              ":>8} | time: {:>8}"
            .format(format_bytes(rss_after - rss_before),
                    format_bytes(vms_after - vms_before),
                    format_bytes(shared_after - shared_before),
                    elapsed_time))
    return(dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts, error_FWHMx, error_FWHMy)


def check_registration(img0, img1, **kwargs):
    '''
    Debugging tool. Perform SIFT registration check on two images. Will perform SIFT, and then report Residual Errors and plot residual error histograms
    
    Parameters:
    ---------
    img0 : 2D array
    img1 : 2D array
    
    kwargs:
    ---------
    thr_min : float
        CDF threshold for determining the minimum data value.
    thr_max : float
        CDF threshold for determining the maximum data value.
    nbins : int
        Number of histogram bins for building the PDF and CDF. Default is 256.
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
        Note that the its meaning is different from the contrastThreshold,
        i.e. the larger the edgeThreshold, the less features are filtered out
        (more features are retained).
    SIFT_sigma : double
        SIFT library default is 1.6.  The sigma of the Gaussian applied to the input image at the octave #0.
        If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
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
    targ_vector = 1D float array
        target vector for regularization
    solver : str
        Solver used for SIFT ('RANSAC' or 'LinReg')
    RANSAC_initial_fraction : float
        Fraction of data points for initial RANSAC iteration step. Default is 0.005.
    Lowe_Ratio_Threshold : float
        Threshold for Lowe's Ratio Test. Default is 0.7.
    drmax : float
        In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
        In the case of 'LinReg' - outlier threshold for iterative regression.
    max_iter : int
        Max number of iterations in the iterative procedure above (RANSAC or LinReg). Default is 1000.
    save_res_png  : boolean
        Save PNG images of the intermediate processing statistics and final registration quality check.
    save_filename : string
        Filename for saving the images.
    dpi : int
        DPI for PNG output. Default is 300.
    Sample_ID : string
        Sample_ID. Used for comment in the PNG output.
    filename : String
        Filename. Used for comment in the PNG output.
    verbose : boolean
        If True, outputs will be printed.
    fontsize : int
        Fontsize. Default is 12.
    fsize_text : int
        Secondary Fontsize. Default is 6.
    Returns:
        return error_FWHMx, error_FWHMy
    '''
    thr_min = kwargs.get("thr_min", 1e-3)
    thr_max = kwargs.get("thr_max", 1e-3)
    nbins = kwargs.get('nbins', 256)
    SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
    SIFT_nOctaveLayers = kwargs.get('SIFT_nOctaveLayers', 3)
    SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 10.0)
    SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.025)
    SIFT_sigma = kwargs.get('SIFT_sigma', 1.6)

    TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
    l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
    l2_matrix_default = np.eye(6)*l2_param_default                   # initially set equal shrinkage on all coefficients
    l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
    l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
    l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
    targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
    solver = kwargs.get("solver", 'RANSAC')
    RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.005)  # fraction of data points for initial RANSAC iteration step.
    Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)   # threshold for Lowe's Ratio Test
    drmax = kwargs.get("drmax", 1.5)
    max_iter = kwargs.get("max_iter", 2500)
    save_res_png  = kwargs.get("save_res_png", True)
    save_filename = kwargs.get('save_filename', 'check_registration_output.png')
    dpi = kwargs.get('dpi', 300)
    
    Sample_ID = kwargs.get('Sample_ID', '')
    filename = kwargs.get('filename', '') 
    verbose = kwargs.get('verbose', True)
    fontsize = kwargs.get('fontsize', 12) 
    fsize_text = kwargs.get('fsize_text', 6)
    
    YResolution, XResolution = img0.shape
    
    SIFT_kwargs = {
        'thr_min' : thr_min, 
        'thr_max' : thr_max,
        'nbins' : nbins,
        'TransformType' : TransformType,
        'SIFT_nfeatures' : SIFT_nfeatures,
        'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
        'SIFT_contrastThreshold' : SIFT_contrastThreshold,
        'SIFT_edgeThreshold' : SIFT_edgeThreshold,
        'SIFT_sigma' : SIFT_sigma,
        'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
        'l2_matrix' : l2_matrix,
        'targ_vector': targ_vector, 
        'solver' : 'RANSAC',
        'RANSAC_initial_fraction' : RANSAC_initial_fraction,
        'drmax' : drmax,
        'max_iter' : max_iter}
    
    d0, d1 = get_min_max_thresholds(img0, thr_min=thr_min, thr_max=thr_max, disp_res=False)
    img0_uint8 = np.clip(255*(img0-d0)/(d1-d0), 0, 255).astype(np.uint8)
    d0, d1 = get_min_max_thresholds(img1, thr_min=thr_min, thr_max=thr_max, disp_res=False)
    img1_uint8 = np.clip(255*(img1-d0)/(d1-d0), 0, 255).astype(np.uint8)
    if verbose:
        print('thr_min={:.0e}, thr_max={:.0e}'.format(thr_min, thr_max))
        print(TransformType.__name__)
        print('SIFT_nfeatures={:d}'.format(SIFT_nfeatures))
        print('SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold))
        print('SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma))
        print('RANSAC_initial_fraction={:.3f}'.format(RANSAC_initial_fraction))
        print('drmax={:.3f}'.format(drmax))
        
        print('')
    if verbose:
        print('Extracting KeyPoints (SIFT) on the first image')
    sift1 = cv2.SIFT_create(nfeatures=SIFT_nfeatures, nOctaveLayers=SIFT_nOctaveLayers, edgeThreshold=SIFT_edgeThreshold, contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma)
    kp1, des1 = sift1.detectAndCompute(img0_uint8, None)
    n_kpts=len(kp1)
    if verbose:
        print('Extracted {:d} keypoints'.format(n_kpts))
    print('Extracting KeyPoints (SIFT) on the second image')
    sift2 = cv2.SIFT_create(nfeatures=SIFT_nfeatures, nOctaveLayers=SIFT_nOctaveLayers, edgeThreshold=SIFT_edgeThreshold, contrastThreshold=SIFT_contrastThreshold, sigma=SIFT_sigma)
    kp2, des2 = sift2.detectAndCompute(img1_uint8, None)
    if verbose:
        print('Extracted {:d} keypoints'.format(len(kp2)))

    if verbose:
        print('Finding Matches')
    if TransformType == RegularizedAffineTransform:
        def estimate(self, src, dst):
            self.params = determine_regularized_affine_transform(src, dst, l2_matrix, targ_vector)
        RegularizedAffineTransform.estimate = estimate
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's Ratio test
    good = []
    for m, n in matches:
        if m.distance < Lowe_Ratio_Threshold * n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 2)

    if verbose:
        print('Finding Transformation Matrix')
    min_samples = np.int32(len(src_pts)*RANSAC_initial_fraction)
    model, inliers = ransac((src_pts, dst_pts),
        TransformType, min_samples = min_samples,
        residual_threshold = drmax, max_trials = max_iter)
    n_inliers = np.sum(inliers)
    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in src_pts[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in dst_pts[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]
    src_pts_ransac = np.float32([ inlier_keypoints_left[m.queryIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    dst_pts_ransac = np.float32([ inlier_keypoints_right[m.trainIdx].pt for m in placeholder_matches ]).reshape(-1, 2)
    kpts = [src_pts_ransac, dst_pts_ransac]
    transform_matrix = model.params
    if verbose:
        print('Transform_Matrix:')
        print(transform_matrix)

    iteration = len(src_pts)- len(src_pts_ransac)
    reg_errors, xshifts, yshifts = estimate_kpts_transform_error(src_pts_ransac, dst_pts_ransac, transform_matrix)
    error_abs_mean = np.mean(np.abs(reg_errors))
    
    fig_xsize = 10.0
    fig_ysize = fig_xsize * YResolution / XResolution + 5.0
    fig = plt.figure(figsize=(fig_xsize, fig_ysize))
    fig.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.99, wspace=0.15, hspace=-0.10)
    gs = GridSpec(3, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0:2, :])
    ax0.imshow(img1, cmap='Greys', vmin = d0, vmax=d1)
    ax0.axis(False)

    n_matches = len(xshifts)
    if n_matches > 0:
        x, y = dst_pts_ransac.T
        M = np.sqrt(xshifts*xshifts+yshifts*yshifts)
        xs = xshifts
        ys = yshifts
        # the code below is for vector map. vectors have origin coordinates x and y, and vector projections xs and ys.
        vec_field = ax0.quiver(x,y,xs,ys,M, scale=50, width =0.0015, cmap='jet')
        cbar = fig.colorbar(vec_field, pad=0.015, shrink=0.70, orientation = 'horizontal', format="%.1f")
        cbar.set_label('Magnitude of the Residual Registration Error (pix)', fontsize=fontsize)

        ax0.text(0.005, 1.00 - 0.010*XResolution/YResolution, filename, fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.023*XResolution/YResolution, Sample_ID, fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.036*XResolution/YResolution, 'thr_min={:.0e}, thr_max={:.0e}'.format(thr_min, thr_max), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.049*XResolution/YResolution, TransformType.__name__, fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.062*XResolution/YResolution, 'SIFT_nfeatures={:d}'.format(SIFT_nfeatures), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.075*XResolution/YResolution, 'SIFT_nOctaveLayers={:d},  SIFT_edgeThreshold={:.3f}'.format(SIFT_nOctaveLayers, SIFT_edgeThreshold), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.088*XResolution/YResolution, 'SIFT_contrastThreshold={:.3f},  SIFT_sigma={:.3f}'.format(SIFT_contrastThreshold, SIFT_sigma), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.101*XResolution/YResolution, 'RANSAC_initial_fraction={:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.114*XResolution/YResolution,  'drmax={:.3f}'.format(drmax), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.005, 1.00 - 0.127*XResolution/YResolution, '# of keypoints = {:d}, # of matches ={:d}'.format(n_kpts, n_matches), fontsize=fsize_text, transform=ax0.transAxes)
        
        ax0.text(0.75, 1.00 - 0.023*XResolution/YResolution, 'Sxx={:.5f},  Sxy={:.5f},  Tx={:.3e}'.format(transform_matrix[0,0], transform_matrix[0,1], transform_matrix[0,2]), fontsize=fsize_text, transform=ax0.transAxes)
        ax0.text(0.75, 1.00 - 0.036*XResolution/YResolution, 'Syx={:.5f},  Syy={:.5f},  Ty={:.3e}'.format(transform_matrix[1,0], transform_matrix[1,1], transform_matrix[1,2]), fontsize=fsize_text, transform=ax0.transAxes)
    if verbose:
        print('RANSAC_initial_fraction = {:.4f}, max_iter={:d}'.format(RANSAC_initial_fraction, max_iter))
        print('# of keypoints = {:d}, # of matches ={:d}'.format(n_kpts, n_matches))

    axx = fig.add_subplot(gs[2, 0])
    xcounts, xbins, xhist_patches = axx.hist(xshifts, bins=64)
    error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, max_aver_aperture=5)
    dbx = (xbins[1]-xbins[0])/2.0
    axx.set_xlabel('Residual X Error (pixels)')
    axx.set_ylabel('Count')
    axy = fig.add_subplot(gs[2, 1])
    ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64)
    error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, max_aver_aperture=5)
    dby = (ybins[1]-ybins[0])/2.0
    axy.set_xlabel('Residual Y Error (pixels)')
    axy.set_ylabel('Count')

    xcounts, xbins, xhist_patches = axx.hist(xshifts, bins=64, color='#1f77b4')
    error_FWHMx, indxi, indxa, mxx, mxx_ind = find_FWHM(xbins, xcounts[:-1], verbose=False, max_aver_aperture=5)
    dbx = (xbins[1]-xbins[0])/2.0
    #axx.plot([xbins[indxi]+dbx, xbins[indxa]+dbx], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
    axx.plot([xbins[indxi], xbins[indxa]], [mxx/2.0, mxx/2.0], 'r', linewidth = 4)
    axx.plot([xbins[mxx_ind]+dbx], [mxx], 'rd')
    axx.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(xshifts)), transform=axx.transAxes, fontsize=fontsize)
    axx.text(0.05, 0.8, 'median={:.3f}'.format(np.median(xshifts)), transform=axx.transAxes, fontsize=fontsize)
    axx.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMx), transform=axx.transAxes, fontsize=fontsize)
    ycounts, ybins, yhist_patches = axy.hist(yshifts, bins=64, color='#1f77b4')
    error_FWHMy, indyi, indya, mxy, mxy_ind = find_FWHM(ybins, ycounts[:-1], verbose=False, max_aver_aperture=5)
    dby = (ybins[1]-ybins[0
        ])/2.0
    #axy.plot([ybins[indyi] + dby, ybins[indya] + dby], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
    axy.plot([ybins[indyi], ybins[indya]], [mxy/2.0, mxy/2.0], 'r', linewidth = 4)
    axy.plot([ybins[mxy_ind] + dby], [mxy], 'rd')
    axy.text(0.05, 0.9, 'mean={:.3f}'.format(np.mean(yshifts)), transform=axy.transAxes, fontsize=fontsize)
    axy.text(0.05, 0.8, 'median={:.3f}'.format(np.median(yshifts)), transform=axy.transAxes, fontsize=fontsize)
    axy.text(0.05, 0.7, 'FWHM={:.3f}'.format(error_FWHMy), transform=axy.transAxes, fontsize=fontsize)

    
    for ax in [axx, axy]:
        ax.grid(True)
        
    if save_res_png:
        axx.text(-0.07, -0.13, save_filename, transform=axx.transAxes, fontsize=fontsize-2)
        if verbose:
            print('Figure is saved into the filr: ', save_filename) 
        fig.savefig(save_filename, dpi=dpi)
    return error_FWHMx, error_FWHMy


def save_inlens_data(fname):
    tfr = FIBSEM_frame(fname)
    tfr.save_images_tif('A')
    return fname


def transform_chunk_of_frames(frame_filenames, xsz, ysz, ftype,
                        flatten_image, image_correction_file,
                        perform_transformation, tr_matrices, shift_matrix, inv_shift_matrix,
                        xi, xa, yi, ya,
                        ImgB_fraction=0.0,
                        invert_data=False,
                        int_order=1,
                        flipY=False,
                        fill_value=0.0):
    '''
    Transform Chunk of Frames and average into a single transformed frames. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com

    Parameters
    frame_filenames : list of strings
        Filenames (Full paths) of FIB-SEM frame files for every frame in frame_inds
    xsz  :  int
        X-size (pixels)
    ysz  :  int
        Y-size (pixels)
    ftype : int
        File Type. 0 for Shan's .dat files, 1 for tif files
    flatten_image : boolean
        perform image flattening
    image_correction_file : str
        full path to a binary filename that contains source name (image_correction_source) and correction array (img_correction_array)
    xi : int
        Low X-axis bound for placing the transformed frame into the image before transformation
    xa : int
        High X-axis bound for placing the transformed frame into the image before transformation
    yi : int
        Low Y-axis bound for placing the transformed frame into the image before transformation
    ya : int
        High Y-axis bound for placing the transformed frame into the image before transformation
    perform_transformation  : boolean
        perform transformation
    tr_matrices : list of 2D (or 3d array)
        Transformation matrix for every frame in frame_inds. Each transformation matrix is in a form:
            [[Sxx Sxy Tx]
            [Syx  Syy Ty]
            [0    0   1]]
    shift_matrix : 2d array
        shift matrix
    inv_shift_matrix : 2d array
        inverse shift matrix.
    ImgB_fraction : float
        Fractional weight of Image B for fused images, default is 0
    invert_data : boolean
        Invert data, default is False.
    int_order : int
        Default is 1. Interpolation order (0: Nearest-neighbor, 1: Bi-linear (default), 2: Bi-quadratic, 3: Bi-cubic, 4: Bi-quartic, 5: Bi-quintic)
    flipY : boolean
        Flip output along Y-axis, default is False.
    fill_value : float
        fill value for padding. Default is zero.
    
    Returns
    '''
    transformed_img = np.zeros((ysz, xsz), dtype=float)
    num_frames = len(frame_filenames)

    for frame_filename, tr_matrix in zip(frame_filenames, tr_matrices):
        #frame_img = np.zeros((ysz, xsz), dtype=float) + fill_value
        frame_img = np.full((ysz, xsz), fill_value, dtype=float)
        frame = FIBSEM_frame(frame_filename, ftype=ftype, calculate_scaled_images=False)

        if ImgB_fraction < 1e-5:
            #image = frame.RawImageA.astype(float)
            if flatten_image:
                image = (frame.flatten_image(image_correction_file = image_correction_file)[0]).astype(float)
            else:
                image = frame.RawImageA.astype(float)
        else:
            if flatten_image:
                flattened_images = frame.flatten_image(image_correction_file = image_correction_file)
                flattened_RawImageA = flattened_images[0].astype(float)
                if len(flattened_images)>1:
                    flattened_RawImageB = flattened_images[1].astype(float)
                else:
                    flattened_RawImageB = frame.RawImageB.astype(float)
                image = flattened_RawImageA* (1.0 - ImgB_fraction) + flattened_RawImageB * ImgB_fraction
            else:
                image = frame.RawImageA.astype(float) * (1.0 - ImgB_fraction) + frame.RawImageB.astype(float) * ImgB_fraction

        if invert_data:
            frame_img[yi:ya, xi:xa] = np.negative(image)
            '''
            if frame.EightBit==0:
                frame_img[yi:ya, xi:xa] = np.negative(image)
            else:
                frame_img[yi:ya, xi:xa]  =  np.uint8(255) - image
            '''
        else:
            frame_img[yi:ya, xi:xa]  = image

        if perform_transformation:
            transf = ProjectiveTransform(matrix = shift_matrix @ (tr_matrix @ inv_shift_matrix))
            frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True, mode='constant', cval=fill_value)
        else:
            frame_img_reg = frame_img.copy()

        transformed_img = transformed_img + frame_img_reg
        
    if num_frames > 1:
        transformed_img = transformed_img/num_frames

    if flipY:
        transformed_img = np.flip(transformed_img, axis=0)
    '''
    if frame.EightBit==1:
        transformed_img = np.clip(np.round(transformed_img) , 0, 255)
    '''

    return transformed_img
    

def transform_and_save_chunk_of_frames(chunk_of_frame_parametrs):
    '''
    Transform Chunk of Frames and save into a single transformed frame. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com

    Parameters
    chunk_of_frame_parametrs : list of following parameters
        [save_filename, frame_filenames, tr_matrices, deformation_fields, image_scales, image_offsets, tr_args]
    
    save_filename : path
        Filename for saving the transformed frame
    frame_filenames : list of strings
        Filenames (Full paths) of FIB-SEM frame files for every frame in frame_inds
    tr_matrices : list of 2D (or 3D array)
        Transformation matrix for every frame in frame_inds. Each transformation matrix is in a form:
            [[Sxx Sxy Tx]
            [Syx  Syy Ty]
            [0    0   1]]
    deformation_fields : list of 1D or 2D arrays (or 2D or 3D array)
        Deformation Field for every frame in frame_inds
    image_scales : list (array) of floats
        image multipliers for image rescaling: I = (I-image_offset)*image_scale + image_offset
    image_offsets : list (array) of floats
        image offsets for image rescaling: I = (I-image_offset)*image_scale + image_offset

    tr_args : list of lowwowing parameters:
        tr_args = [ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, flipY, flatten_image, image_correction_file, perform_transformation, shift_matrix, inv_shift_matrix, perform_deformation, deformation_type, ftype, dtp, fill_value]
    
    ImgB_fraction : float
        Fractional weight of Image B for fused images, default is 0
    xsz  :  int
        X-size (pixels)
    ysz  :  int
        Y-size (pixels)
    xi : int
        Low X-axis bound for placing the transformed frame into the image before transformation
    xa : int
        High X-axis bound for placing the transformed frame into the image before transformation
    yi : int
        Low Y-axis bound for placing the transformed frame into the image before transformation
    ya : int
        High Y-axis bound for placing the transformed frame into the image before transformation
    int_order : int
        Default is 1. Interpolation order (0: Nearest-neighbor, 1: Bi-linear (default), 2: Bi-quadratic, 3: Bi-cubic, 4: Bi-quartic, 5: Bi-quintic)
    invert_data : boolean
        Invert data, default is False.
    flipY : boolean
        Flip output along Y-axis, default is False.
    flatten_image : boolean
        perform image flattening
    image_correction_file : str
        full path to a binary filename that contains source name (image_correction_source) and correction array (img_correction_array)
    perform_transformation  : boolean
        perform transformation     
    shift_matrix : 2d array
        shift matrix
    inv_shift_matrix : 2d array
        inverse shift matrix.
    perform_deformation : boolean
        If True - the data is deformed (in addition to tyransformation defined above) using the deformation field data defined below
    deformation_type : str
        Options are:
            'post_1DY'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
            'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
    ftype : int
        File Type. 0 for Shan's .dat files, 1 for tif files
    dtp : data type
        Python data type for saving. Default is int16, the other option currently is uint8.
    verbose : boolean 

    Returns
    '''
    #save_filename, frame_filenames, tr_matrices, image_scales, image_offsets, tr_args = chunk_of_frame_parametrs
    save_filename, frame_filenames, tr_matrices, deformation_fields, image_scales, image_offsets, tr_args = chunk_of_frame_parametrs
    #ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, flipY, flatten_image, image_correction_file, perform_transformation, shift_matrix, inv_shift_matrix, ftype, dtp, fill_value = tr_args
    ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, flipY, flatten_image, image_correction_file, perform_transformation, shift_matrix, inv_shift_matrix, perform_deformation, deformation_type, ftype, dtp, fill_value = tr_args
    num_frames = len(frame_filenames)
    transformed_img = np.zeros((ysz, xsz), dtype=float)
    verbose = False
    
    for frame_filename, tr_matrix, deformation_field, image_scale, image_offset in zip(frame_filenames, tr_matrices, deformation_fields, image_scales, image_offsets):
        #frame_img = np.zeros((ysz, xsz), dtype=float) + fill_value
        frame_img = np.full((ysz, xsz), fill_value, dtype=float)
        frame = FIBSEM_frame(frame_filename, ftype=ftype, calculate_scaled_images=False)

        if ImgB_fraction < 1e-5:
            #image = frame.RawImageA.astype(float)
            if flatten_image:
                image = (frame.flatten_image(image_correction_file = image_correction_file)[0]).astype(float)
            else:
                image = frame.RawImageA.astype(float)
        else:
            if flatten_image:
                flattened_images = frame.flatten_image(image_correction_file = image_correction_file)
                flattened_RawImageA = flattened_images[0].astype(float)
                if len(flattened_images)>1:
                    flattened_RawImageB = flattened_images[1].astype(float)
                else:
                    flattened_RawImageB = frame.RawImageB.astype(float)
                image = flattened_RawImageA* (1.0 - ImgB_fraction) + flattened_RawImageB * ImgB_fraction
            else:
                image = frame.RawImageA.astype(float) * (1.0 - ImgB_fraction) + frame.RawImageB.astype(float) * ImgB_fraction
        image = (image - image_offset) * image_scale + image_offset

        if invert_data:
            frame_img[yi:ya, xi:xa] = np.negative(image)
            '''
            if frame.EightBit==0:
                frame_img[yi:ya, xi:xa] = np.negative(image)
            else:
                frame_img[yi:ya, xi:xa]  =  np.uint8(255) - image
            '''
        else:
            frame_img[yi:ya, xi:xa]  = image

        if perform_transformation:
            if perform_deformation:
                #df = convert_tr_matr_into_deformation_field(tr_matrix, image.shape).astype(np.float32)
                df = convert_tr_matr_into_deformation_field(shift_matrix @ (tr_matrix @ inv_shift_matrix), frame_img.shape).astype(np.float32)
                '''
                int_order:   #  The order of interpolation. The order has to be in the range 0-5:
                             #    - 0: Nearest-neighbor
                             #    - 1: Bi-linear (default)
                             #    - 2: Bi-quadratic
                             #    - 3: Bi-cubic
                             #    - 4: Bi-quartic
                             #    - 5: Bi-quintic
                '''
                if int_order == 0:
                    interpolation = cv2.INTER_NEAREST
                if int_order == 1:
                    interpolation = cv2.INTER_LINEAR
                if int_order == 3:
                    interpolation = cv2.INTER_CUBIC

                if deformation_type == 'prior_1DY':
                    pass
                if deformation_type == 'prior_1DX':
                    pass
                if deformation_type == 'prior_2D':
                    pass
                if deformation_type == 'post_1DY':
                    if verbose:
                        print('Performing Transformation using CV2.remap with additional post_1DY deformation fields')
                    orig_shape = frame.RawImageA.shape
                    additional_deformation = np.repeat(deformation_field[:, np.newaxis], orig_shape[1], 1).astype(np.float32)
                    xi_loc, yi_loc, padx_loc, pady_loc  = determine_pad_offsets(orig_shape, tr_matrix[np.newaxis, :, :])
                    if verbose:
                        print('Global: xi={:d}, yi={:d}. Local xi_loc={:d}, yi_loc={:d}'.format(xi, yi, xi_loc, yi_loc))
                    additional_deformation_padded = np.pad(additional_deformation, pad_width=((yi_loc, (ysz-orig_shape[0]-yi_loc)), (xi_loc, (xsz-orig_shape[1]-xi_loc))), mode='edge')
                    df[:, :, 0] = df[:, :, 0] + additional_deformation_padded
                if deformation_type == 'post_1DX':
                    pass
                if deformation_type == 'post_2D':
                    pass
                frame_img_reg = cv2.remap(frame_img, df[:, :, 0].astype(np.float32), df[:, :, 1].astype(np.float32), interpolation=interpolation, borderValue=fill_value )
            else:
                transf = ProjectiveTransform(matrix = shift_matrix @ (tr_matrix @ inv_shift_matrix))
                frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True, mode='constant', cval=fill_value)
        else:
            frame_img_reg = frame_img.copy()

        transformed_img = transformed_img + frame_img_reg
        
    if num_frames > 1:
        transformed_img = transformed_img/num_frames

    if flipY:
        transformed_img = np.flip(transformed_img, axis=0)

    try:
        tiff.imsave(save_filename, transformed_img.astype(dtp))
    except:
        tiff.imwrite(save_filename, transformed_img.astype(dtp))

    return save_filename


def analyze_registration_frames(DASK_client, frame_filenames, **kwargs):
    '''
    Transform and save FIB-SEM data set. A new vesion, with variable zbin_factor option. ©G.Shtengel 09/2022 gleb.shtengel@gmail.com

    Parameters
    DASK_client : 
    frame_filenames : list of strings
        List of filenames (one for each transformed / z-binned frame)
    
    kwargs
    ---------
    use_DASK : boolean
        perform remote DASK computations
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    save_registration_summary : boolean
        If True (default), the rgistration analysis will be saved
    data_dir : str
        data directory (path)
    frame_inds : int array
        Array of frame indices. If not set or set to np.array((-1)), all frames will be analyzed
    fnm_reg : str
        filename for the final registered dataset
    npts : array or list of int
        Numbers of Keypoints used for registration
    error_abs_mean : array or list of float
        mean abs error between registered key-points
    eval_bounds : list of [xi_eval, xa_eval, yi_eval, ya_eval] lists of int
        Evaluation boundaries for analysis
    eval_metrics : list of str
        list of evaluation metrics to use. default is ['NSAD', 'NCC', 'NMI', 'FSC']
    save_sample_frames_png : boolean
        If True, sample frames with superimposed eval box and registration analysis data will be saved into png files
    sample_frame_inds : list of int
        list of sample frame indices
    save_registration_summary : boolean
        If True, the registration summary is saved into XLSX file
    disp_res : boolean
        If True (default), intermediate messages and results will be displayed.

    Returns:
    reg_summary, reg_summary_xlsx
        reg_summary : pandas DataFrame
        reg_summary = pd.DataFrame(np.vstack((npts, error_abs_mean, image_nsad, image_ncc, image_mi)
        reg_summary_xlsx : name of the XLSX spreadsheet file containing the data
    '''

    use_DASK = kwargs.get("use_DASK", True)  # do not use DASK the data is to be saved
    DASK_client_retries = kwargs.get("DASK_client_retries", 3)
    
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    fpath_reg = os.path.join(data_dir, fnm_reg)

    save_sample_frames_png = kwargs.get("save_sample_frames_png", True)
    disp_res = kwargs.get("disp_res", True)
    save_registration_summary = kwargs.get('save_registration_summary', True)
    dump_filename = kwargs.get('dump_filename', '')

    frame_inds_default = np.arange(len(frame_filenames))
    frame_inds = np.array(kwargs.get("frame_inds", frame_inds_default))
    nfrs = len(frame_inds)                                                   # number of source images(frames) before z-binning
    sample_frame_inds = kwargs.get("sample_frame_inds", [frame_inds[nfrs//10], frame_inds[nfrs//2], frame_inds[nfrs//10*9-1]])
    npts = kwargs.get("npts", False)
    error_abs_mean = kwargs.get("error_abs_mean", False)

    first_frame = tiff.imread(frame_filenames[frame_inds[0]])
    ya, xa = first_frame.shape
    eval_bounds = kwargs.get("eval_bounds", [[0, xa, 0, ya]]*nfrs)
    eval_metrics = kwargs.get('eval_metrics', ['NSAD', 'NCC', 'NMI', 'FSC'])

    params_frames = []
    for j, frame_ind in enumerate(tqdm(frame_inds[0:-1], desc='Setting up parameter sets', display=False)):
        params_frames.append([frame_filenames[frame_ind], frame_filenames[frame_ind+1], eval_bounds[j], eval_metrics])

    if use_DASK:
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Analyzing Frame Registration: Starting DASK jobs')
        futures_til = DASK_client.map(Two_Image_Analysis, params_frames, retries = DASK_client_retries)
        results_til = DASK_client.gather(futures_til)
        image_metrics = np.array(results_til)  # 2D array  np.array([[image_nsad, image_ncc, image_mi]])
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Finished DASK jobs')

    else:   # if DASK is not used - perform local computations
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Analyzing Frame Registration: Will perform local computations')
        image_metrics = np.zeros((nfrs-1, len(eval_metrics)), dtype=float)
        for j, params_frame in enumerate(tqdm(params_frames, desc = 'Analyzing frame pairs', display = disp_res)):
            image_metrics[j, :] = Two_Image_Analysis(params_frame)
        results_til = []
     
    # save sample frames
    if save_sample_frames_png:
        for frame_ind in sample_frame_inds:
            filename_frame_png = os.path.splitext(fpath_reg)[0]+'_sample_image_frame{:d}.png'.format(frame_ind)
            fr_img = tiff.imread(frame_filenames[frame_ind]).astype(float)
            yshape, xshape = fr_img.shape
            fig, ax = plt.subplots(1,1, figsize=(3.0*xshape/yshape, 3))
            fig.subplots_adjust(left=0.0, bottom=0.00, right=1.0, top=1.0)
            xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds[frame_ind]
            dmin, dmax = get_min_max_thresholds(fr_img[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
            ax.imshow(fr_img, cmap='Greys', vmin=dmin, vmax=dmax)
            frame_text = 'Frame={:d}'.format(frame_ind)
            ax.text(0.02, 0.95, frame_text, color='red', transform=ax.transAxes, fontsize=11)
            for k, metric in enumerate(eval_metrics):
                sample_text = metric + '={:.3f}'.format(image_metrics[frame_ind, k])
                ax.text(0.50, 0.95-k*0.1, sample_text, color='red', transform=ax.transAxes, fontsize=11)
            rect_patch = patches.Rectangle((xi_eval, yi_eval), np.abs(xa_eval-xi_eval)-2, np.abs(ya_eval-yi_eval)-2, linewidth=1.0, edgecolor='yellow',facecolor='none')
            ax.add_patch(rect_patch)
            ax.axis('off')
            fig.savefig(filename_frame_png, dpi=300, bbox_inches='tight', transparent=True, pad_inches=0)   # save the figure to file
            plt.close(fig)

    columns = ['Frame', 'xi_eval', 'xa_eval', 'yi_eval', 'ya_eval'] + eval_metrics
    reg_summary = pd.DataFrame(np.vstack((frame_inds[1:].T, np.array(eval_bounds)[frame_inds[1:], :].T, np.array(image_metrics).T)).T, columns = columns, index = None)
    if npts:
        reg_summary['Npts'] = npts
        columns = columns + ['Npts']
    if error_abs_mean:
        reg_summary['Mean Abs Error'] = error_abs_mean
        columns = columns + ['Mean Abs Error']
    
    if save_registration_summary:
        registration_summary_xlsx = fpath_reg.replace('.mrc', '_RegistrationQuality.xlsx')
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the Registration Quality Statistics into the file: ', registration_summary_xlsx)
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        xlsx_writer = pd.ExcelWriter(registration_summary_xlsx, engine='xlsxwriter')
        reg_summary.to_excel(xlsx_writer, index=None, sheet_name='Registration Quality Statistics')
        Stack_info = pd.DataFrame([{'Stack Filename' : fnm_reg, 'dump_filename' : dump_filename}]).T # prepare to be save in transposed format
        try:
            del kwargs['eval_bounds']
        except:
            pass
        SIFT_info = pd.DataFrame([kwargs]).T   # prepare to be save in transposed format
        #Stack_info = Stack_info.append(SIFT_info)  append has been removed from pandas as of 2.0.0, use concat instead
        Stack_info = pd.concat([Stack_info, SIFT_info], axis=1)
        Stack_info.to_excel(xlsx_writer, header=False, sheet_name='Stack Info')
        #xlsx_writer.save()
        xlsx_writer.close()
    else:
        registration_summary_xlsx = 'Registration data not saved'

    return reg_summary, registration_summary_xlsx


def transform_and_save_frames(DASK_client, frame_inds, fls, tr_matr_cum_residual, **kwargs):
    '''
    Transform and save FIB-SEM data set. A new vesion, with variable zbin_factor option. ©G.Shtengel 01/2023 gleb.shtengel@gmail.com

    Parameters
    DASK_client : DASK client
    frame_inds : int array
        Array of frame indices. If not set or set to np.array((-1)), all frames will be transformed
    fls : array of strings
        full array of filenames
    tr_matr_cum_residual : array
        transformation matrix
    
    kwargs
    ---------
    use_DASK : boolean
        perform remote DASK computations
    DASK_client_retries : int
        Number of allowed automatic retries if a task fails. Default is 3.
    ftype : int
        file type (0 - Shan Xu's .dat, 1 - tif)
    data_dir : str
        data directory (path)
    XResolutions : array
        X-sizes of the frames
    YResolutions : array
        Y-sizes of the frames
    ImgB_fraction : float
        fractional ratio of Image B to be used for constructing the fused image:
        ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
    pad_edges : boolean
        If True, the data will be padded before transformation to avoid clipping.
    flipY : boolean
        If True, the data will be flipped along Y-axis. Default is False.
    zbin_factor : int
        binning factor along Z-axis
    perform_transformation : boolean
        If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
    int_order : int
        The order of interpolation. 1: Bi-linear
    flatten_image : boolean
        perform image flattening
    image_correction_file : str
        full path to a binary filename that contains source name (image_correction_source) and correction array (img_correction_array)
    image_scales : array of floats
        image multipliers for image rescaling: I = (I-image_offset)*image_scale + image_offset
    image_offsets : array of floats
        image offsets for image rescaling: I = (I-image_offset)*image_scale + image_offset
    invert_data : boolean
        If True - the data is inverted.
    perform_deformation : boolean
        If True - the data is deformed (in addition to tyransformation defined above) using the deformation field data defined below
    deformation_type : str
        Options are:
            'post_1DY'  - Default. Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
            'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
            'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
            'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
    deformation_fields : float array
    dtp  : dtype
        Python data type for saving. Default is int16.
    fill_value : float
        Fill value for padding. Default is zero.
    disp_res : boolean
        Default is False.

    Returns:
    registered_filenames : list of filenames (one for each transformed / z-binned frame)
    
    '''
    ftype = kwargs.get("ftype", 0)
    fill_value = kwargs.get('fill_value', 0.0)
    test_frame = FIBSEM_frame(fls[0], ftype=ftype, calculate_scaled_images=False)
    
    save_transformed_dataset = kwargs.get("save_transformed_dataset", True)
    use_DASK = kwargs.get("use_DASK", False)  # do not use DASK the data is to be saved
    DASK_client_retries = kwargs.get("DASK_client_retries", 3)
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')
    dump_filename = kwargs.get('dump_filename', '')
    ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)         # fusion fraction. In case if Img B is present, the fused image 
                                                            # for each frame will be constructed ImgF = (1.0-ImgB_fraction)*ImgA + ImgB_fraction*ImgB
    if test_frame.DetB == 'None':
        ImgB_fraction=0.0

    XResolutions = kwargs.get("XResolutions", np.full(len(fls), test_frame.XResolution))
    YResolutions = kwargs.get("YResolutions", np.full(len(fls), test_frame.YResolution))
    pad_edges =  kwargs.get("pad_edges", True)
    flipY = kwargs.get("flipY", False)
    zbin_factor =  kwargs.get("zbin_factor", 1)
    perform_transformation =  kwargs.get("perform_transformation", True)
    int_order = kwargs.get("int_order", 1)                  # The order of interpolation. 1: Bi-linear
    flatten_image = kwargs.get("flatten_image", False)
    image_correction_file = kwargs.get("image_correction_file", '')
    image_scales = kwargs.get("image_scales", np.full(len(fls), 1.0))
    image_offsets = kwargs.get("image_offsets", np.zeros(len(fls)))
    invert_data =  kwargs.get("invert_data", False)
    perform_deformation = kwargs.get("perform_deformation", False)
    deformation_type = kwargs.get("deformation_type", 'post_1DY')
    dtp = kwargs.get("dtp", np.int16)  # Python data type for saving. Default is int16.
    disp_res = kwargs.get("disp_res", False)
    nfrs = len(frame_inds)                                                   # number of source images(frames) before z-binning
    end_frame = ((frame_inds[0]+len(frame_inds)-1)//zbin_factor+1)*zbin_factor
    st_frames = np.arange(frame_inds[0], end_frame, zbin_factor)             # starting frame for each z-bin
    nfrs_zbinned = len(st_frames)                                            # number of frames after z-ninning
    frames_new = np.arange(nfrs_zbinned-1)
    deformation_fields = kwargs.get('deformation_fields', np.zeros((nfrs, test_frame.YResolution), dtype=float))
    shape = [np.max(YResolutions), np.max(XResolutions)]
    shapes = [YResolutions, XResolutions]
    
    if pad_edges and perform_transformation:
        #Determining padding offsets
        xi, yi, xsz, ysz = determine_sizes_and_offsets(shapes, tr_matr_cum_residual)
        #xi, yi, padx, pady = determine_pad_offsets(shape, tr_matr_cum_residual)
        # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
        # so that the transformed images are not clipped.
        # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
        # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
        # those are calculated below base on the amount of padding calculated above
        shift_matrix = np.array([[1.0, 0.0, xi],
                                 [0.0, 1.0, yi],
                                 [0.0, 0.0, 1.0]]).astype(float)
        inv_shift_matrix = np.linalg.inv(shift_matrix)
    else:
        xi = 0
        yi = 0
        xsz = shape[1]
        ysz = shape[0]
        shift_matrix = np.eye(3,3)
        inv_shift_matrix = np.eye(3,3)
 
    fpath_reg = os.path.join(data_dir, fnm_reg)

    '''
    transform_and_save_chunk_of_frames(save_filename, frame_filenames, tr_matrices, tr_args):
    chunk_of_frame_parametrs = save_filename, frame_filenames, tr_matrices_cum_residual, deformation_fields, image_scales, image_offsets, tr_args
    tr_args = [ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, flipY, flatten_image, image_correction_file, perform_transformation, shift_matrix, inv_shift_matrix, perform_deformation, deformation_type, ftype, dtp, fill_value]
    process_frames = np.arange(st_frame, min(st_frame+zbin_factor, (frame_inds[-1]+1)))
    chunk_of_frame_parametrs_dataset.append([save_filename, process_frames, np.array(tr_matr_cum_residual)[process_frames], tr_args])
    '''
    chunk_of_frame_parametrs_dataset = []

    for j, st_frame in enumerate(tqdm(st_frames, desc='Setting up parameter sets', display=disp_res)):
        xa = xi + XResolutions[st_frame]
        ya = yi + YResolutions[st_frame]
        tr_args = [ImgB_fraction, xsz, ysz, xi, xa, yi, ya, int_order, invert_data, flipY, flatten_image, image_correction_file, perform_transformation, shift_matrix, inv_shift_matrix, perform_deformation, deformation_type, ftype, dtp, fill_value]

        #save_filename = os.path.join(os.path.split(fls[st_frame])[0],'Registered_Frame_{:d}.tif'.format(j))
        save_filename = os.path.splitext(fls[st_frame])[0]+'_transformed.tif'
        process_frames = np.arange(st_frame, min(st_frame+zbin_factor, (frame_inds[-1]+1)))
        if perform_deformation:
            chunk_of_frame_parametrs_dataset.append([save_filename, np.array(fls)[process_frames], np.array(tr_matr_cum_residual)[process_frames], np.array(deformation_fields)[process_frames], np.array(image_scales)[process_frames], np.array(image_offsets)[process_frames], tr_args])
        else:
            chunk_of_frame_parametrs_dataset.append([save_filename, np.array(fls)[process_frames], np.array(tr_matr_cum_residual)[process_frames], np.zeros((len(process_frames), YResolutions[st_frame]), dtype=float), np.array(image_scales)[process_frames], np.array(image_offsets)[process_frames], tr_args])
    
    if use_DASK:
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Transform and Save Chunks of Frames: Starting DASK jobs')
        futures_td = DASK_client.map(transform_and_save_chunk_of_frames, chunk_of_frame_parametrs_dataset, retries = DASK_client_retries)
        registered_filenames = np.array(DASK_client.gather(futures_td))
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Finished DASK jobs')
    else:   # if DASK is not used - perform local computations
        if disp_res:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Transform and Save Chunks of Frames: Will perform local computations')
        registered_filenames = []
        for chunk_of_frame_parametrs in tqdm(chunk_of_frame_parametrs_dataset, desc = 'Transforming and saving frame chunks', display = disp_res):
            registered_filenames.append(transform_and_save_chunk_of_frames(chunk_of_frame_parametrs))

    return registered_filenames


def save_data_stack(FIBSEMstack, **kwargs):
    '''
    Saves the dataset into a file.
    
    Parameters
        FIBSEMstack : 3D array (may be DASK array)
            Data set to be saved
        
    kwargs
    ---------
        data_dir : str
            data directory for saving the data
        fnm_reg : str
            filename for the final registered dataset
        fnm_types : list of strings
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is 'mrc'. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        voxel_size : rec array of 3 elemets
            voxel size in nm
        dtp  : dtype
            Python data type for saving. Default is int16, the other option currently is uint8.
        disp_res : boolean
            Display messages and intermediate results
        chunked_mrc_write : boolean
            if True, the MRC stack is written in chunks, otherwise (False, Default) frame-by-frame
        chunk_length : int
            length of MRC write chunk. Default is 32
        
    Returns:
        fnms_saved : list of strings
            Paths to the files where the data set was saved.
    
    '''
    data_dir = kwargs.get("data_dir", '')
    fnm_reg = kwargs.get("fnm_reg", 'Registered_set.mrc')
    fpath_reg = os.path.join(data_dir, fnm_reg)
    fnm_types = kwargs.get("fnm_types", ['mrc'])
    voxel_size_default = np.rec.array((8.0, 8.0, 8.0), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    voxel_size = kwargs.get("voxel_size", voxel_size_default)
    dtp = kwargs.get("dtp", np.int16)
    disp_res  = kwargs.get("disp_res", False )
    nz, ny, nx = FIBSEMstack.shape
    chunked_mrc_write = kwargs.get('chunked_mrc_write', False)
    chunk_length = kwargs.get('chunk_length', 32)

    if disp_res:
        print('The resulting stack shape will be  nx={:d}, ny={:d}, nz={:d},  data type:'.format(nx, ny, nz), dtp)
        print('Voxel destreak_mrc_stackSize (nm): {:2f} x {:2f} x {:2f}'.format(voxel_size.x, voxel_size.y, voxel_size.z))

    fnms_saved = []
    if len(fnm_types)>0:
        for fnm_type in fnm_types:
            # save dataset at HDF5 file
            if fnm_type == 'h5':
                fpath_reg_h5 = fpath_reg.replace('.mrc', '.h5')
                try:
                    os.remove(fpath_reg_h5)
                except:
                    pass
                fnms_saved.append(fpath_reg_h5)
                if disp_res:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving dataset into Big Data Viewer HDF5 file: ', fpath_reg_h5)
                bdv_writer = npy2bdv.BdvWriter(fpath_reg_h5, nchannels=1, blockdim=((1, 256, 256),))
                bdv_writer.append_view(stack=FIBSEMstack,
                       virtual_stack_dim=(nz,ny,nx),
                       time=0, channel=0, 
                       voxel_size_xyz=(voxel_size.x, voxel_size.y, voxel_size.z),
                       voxel_units='nm')
                bdv_writer.write_xml()
                bdv_writer.close()
            if fnm_type == 'mrc':
                if disp_res:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving dataset into MRC file: ', fpath_reg)
                fnms_saved.append(fpath_reg)
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
                    
                # Make a new, empty memory-mapped MRC file
                if disp_res:
                     print('The resulting stack will saved using mrc_mode={:d},  data type='.format(mrc_mode), dtp)
                mrc = mrcfile.new_mmap(fpath_reg, shape=(nz, ny, nx), mrc_mode=mrc_mode, overwrite=True)
                voxel_size_angstr = voxel_size.copy()
                voxel_size_angstr.x = voxel_size_angstr.x * 10.0
                voxel_size_angstr.y = voxel_size_angstr.y * 10.0
                voxel_size_angstr.z = voxel_size_angstr.z * 10.0
                #mrc.header.cella = voxel_size_angstr
                mrc.voxel_size = voxel_size_angstr
                if chunked_mrc_write:
                    start_frames = np.arange(0, nz, chunk_length)
                    for start_frame in tqdm(start_frames, desc = 'Saving Chunks of Frames into MRC File: ', display = disp_res):
                        stop_frame = np.min((start_frame+chunk_length, nz-1))
                        mrc.data[start_frame:stop_frame] = FIBSEMstack[start_frame:stop_frame].astype(dtp)
                else:
                    for j, FIBSEMframe in enumerate(tqdm(FIBSEMstack, desc = 'Saving Frames into MRC File: ', display = disp_res)):
                        mrc.data[j, :, :] = FIBSEMframe.astype(dtp)
                mrc.close()
    else:
        print('Registered data set is NOT saved into a file')
    return fnms_saved


def check_for_nomatch_frames_dataset(fls, fnms, fnms_matches,
                                     transformation_matrix,
                                     error_abs_mean, npts,
                                     FOVtrend_x, FOVtrend_y,
                                     FIBSEM_Data,
                                     thr_npt, **kwargs):
    data_dir = kwargs.get("data_dir", '')
    ftype = kwargs.get("ftype", 0)
    fnm_reg = kwargs.get("fnm_reg", 'Registration_file.mrc')

    inds_zeros = np.squeeze(np.argwhere(npts < thr_npt ))
    print('Frames with no matches to the next frame:  ', np.array(inds_zeros))
    frames_to_remove = []
    if np.shape(inds_zeros)!=():
        for ind0 in inds_zeros:
            if ind0 < (len(fls)-2) and npts[ind0+1] < thr_npt:
                frames_to_remove.append(ind0+1)
                print('Frame to remove: {:d} : '.format(ind0+1) + ', File: ' + fls[ind0+1])
                frame_to_remove  = FIBSEM_frame(fls[ind0+1], ftype=ftype, calculate_scaled_images=False)
                frame_to_remove.save_snapshot(dpi=300)
        print('Frames to remove:  ', frames_to_remove)

    if len(frames_to_remove) == 0:
        print('No frames selected for removal')
    else:
        # create copies of the original arrays
        fnms_orig = fnms.copy()
        fls_orig = fls.copy()
        error_abs_mean_orig = error_abs_mean.copy()
        tr_matrix_orig = transformation_matrix.copy()

        # go through the frames to be removed and remove the frames from the list and then re-calculate the shift for new neighbours.
        for j,fr in enumerate(tqdm(frames_to_remove, desc = 'Removing frames and finding shifts for new sequential frames')):
            frj = fr-j # to account for the fact that every time we remove a frame the array shrinks and indicis reset
            print('Removing the frame {:d}'.format(frj))
            print(fls[frj])
            fls = np.delete(fls, frj)
            fnms = np.delete(fnms, frj)
            fnms_matches = np.delete(fnms_matches, frj)
            error_abs_mean = np.delete(error_abs_mean, frj)
            FOVtrend_x = np.delete(FOVtrend_x, frj)
            FOVtrend_y = np.delete(FOVtrend_y, frj)
            for j in np.arange(3, len(FIBSEM_Data)): 
                FIBSEM_Data[j] = np.delete(FIBSEM_Data[j], frj, axis = 0)
            transformation_matrix = np.delete(transformation_matrix, frj, axis = 0)
            npts = np.delete(npts, frj, axis = 0)
            fname1 = fnms[frj-1]
            fname2 = fnms[frj]
            new_step4_res = determine_transformations_files([fname1, fname2, kwargs])
            npts[frj-1] = np.array(len(new_step4_res[2][0]))
            error_abs_mean[frj-1] = new_step4_res[3]
            transformation_matrix[frj-1] = np.array(new_step4_res[0])
        print('Mean Number of Keypoints :', np.mean(npts).astype(np.int64))
    return frames_to_remove, fls, fnms, fnms_matches, error_abs_mean, npts, transformation_matrix, FOVtrend_x, FOVtrend_y, FIBSEM_Data

def select_blobs_LoG_analyze_transitions_2D_dataset(params):
    '''
    DASK wrapper for select_blobs_LoG_analyze_transitions
    Finds blobs in the given grayscale image using Laplasian of Gaussians (LoG). gleb.shtengel@gmail.com 06/2023
    
    Parameters:
    params = list of: [fls, frame_ind, ftype, image_name, eval_bounds, offsets_sizes, invert_data, flipY, zbin_factor, perform_transformation, tr_matr_cum_residual, int_order, pad_edges,
        min_sigma, max_sigma, threshold,  overlap, pixel_size, subset_size, bounds, bands, min_thr, transition_low_limit, transition_high_limit, nbins, verbose, disp_res, save_data]
    fls
    frame_ind : index of frame
    ftype
    image_name : str
        Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'
    eval_bounds_single_frame
    offsets = [xi, yi, padx, pady]
    invert_data : boolean
        If True - the data is inverted
    flipY
    zbin_factor
    perform_transformation : boolean
        If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed
    tr_matr_cum_residual
    int_order
    pad_edges : boolean
        If True, the data will be padded before transformation to avoid clipping.
    min_sigma : float
        min sigma (in pixel units) for Gaussian kernel in LoG search.
    max_sigma : float
        min sigma (in pixel units) for Gaussian kernel in LoG search.
    threshold : float
        threshold for LoG search. The absolute lower bound for scale space maxima. Local maxima smaller
        than threshold are ignored. Reduce this to detect blobs with less intensities. 
    overlap : float
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than 'overlap', the smaller blob is eliminated.    
    pixel_size : float
        pixel size in nm.
    subset_size : int
        subset size (pixels) for blob / transition analysis
    bounds : lists
        List of transition limits.
    bands : list of 3 ints
        list of three ints for the averaging bands for determining the left min, peak, and right min of the cross-section profile.
    min_thr : float
        threshold for identifying a 'good' transition (bottom < min_thr* top)
    transition_low_limit : float
        error flag is incremented by 4 if the determined transition distance is below this value.
    transition_high_limit : float
        error flag is incremented by 8 if the determined transition distance is above this value.
    title : str
        title.
    nbins : int
        bins for histogram
    verbose
    disp_res
    save_data
    
    Returns: XY_transitions
        XY_transitions with error_flag=0
    '''
    [fls, frame_ind, ftype, image_name, eval_bounds_single_frame, offsets_sizes, invert_data, flipY, zbin_factor, perform_transformation, tr_matr_cum_residual, int_order, pad_edges, min_sigma, max_sigma, threshold,  overlap, pixel_size, subset_size, bounds, bands, min_thr, transition_low_limit, transition_high_limit, nbins, verbose, disp_res, save_data] = params

    calculate_scaled_images = (image_name == 'ImageA') or (image_name == 'ImageB')
    frame = FIBSEM_frame(fls[frame_ind], ftype=ftype, calculate_scaled_images=calculate_scaled_images)
    shape = [frame.YResolution, frame.XResolution]
    if pad_edges and perform_transformation:
        xi, yi, xsz, ysz = offsets_sizes
        shift_matrix = np.array([[1.0, 0.0, xi],
                                 [0.0, 1.0, yi],
                                 [0.0, 0.0, 1.0]]).astype(float)
        inv_shift_matrix = np.linalg.inv(shift_matrix)
    else:
        xsz = shape[1]
        ysz = shape[0]
        xi = 0
        yi = 0
        shift_matrix = np.eye(3,3)
        inv_shift_matrix = np.eye(3,3)

    xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds_single_frame
    
    frame_img = np.zeros((ysz, xsz), dtype=float)
    frame_eval = np.zeros(((ya_eval-yi_eval), (xa_eval-xi_eval)), dtype=float)

    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Will analyze a subset of ', image_name)

    for j in np.arange(zbin_factor):
        if j>0:
            frame = FIBSEM_frame(fls[frame_ind+j], ftype=ftype, calculate_scaled_images=calculate_scaled_images)
        xa = xi+frame.XResolution
        ya = yi+frame.YResolution
        if invert_data:
            if frame.DetB != 'None':
                if image_name == 'RawImageB':
                    frame_img[yi:ya, xi:xa] = np.negative(frame.RawImageB.astype(float))
                if image_name == 'ImageB':
                    frame_img[yi:ya, xi:xa] = np.negative(frame.ImageB.astype(float))
            if image_name == 'RawImageA':
                frame_img[yi:ya, xi:xa] = np.negative(frame.RawImageA.astype(float))
            else:
                frame_img[yi:ya, xi:xa] = np.negative(frame.ImageA.astype(float))
        else:
            if frame.DetB != 'None':
                if image_name == 'RawImageB':
                    frame_img[yi:ya, xi:xa]  = frame.RawImageB.astype(float)
                if image_name == 'ImageB':
                    frame_img[yi:ya, xi:xa]  = frame.ImageB.astype(float)
            if image_name == 'RawImageA':
                frame_img[yi:ya, xi:xa]  = frame.RawImageA.astype(float)
            else:
                frame_img[yi:ya, xi:xa]  = frame.ImageA.astype(float)

        if perform_transformation:
            transf = ProjectiveTransform(matrix = shift_matrix @ (tr_matr_cum_residual @ inv_shift_matrix))
            frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True)
        else:
            frame_img_reg = frame_img.copy()

        if flipY:
            frame_img_reg = np.flip(frame_img_reg, axis=0)

        frame_eval += frame_img_reg[yi_eval:ya_eval, xi_eval:xa_eval]
    if zbin_factor > 1:
        frame_eval /= zbin_factor
    if verbose:
        print('Subset shape: ', np.shape(frame_eval))

    fname_root = os.path.splitext(os.path.split(fls[frame_ind])[1])[0]+'_'+ image_name +'_'
    fname_base = os.path.split(fls[frame_ind])[0]
    res_png_fname = os.path.join(fname_base, fname_root + 'resolution_results.png')
    examples_png_fname = os.path.join(fname_base, fname_root + 'blob_examples.png')
    results_file_xlsx = os.path.join(fname_base, fname_root + 'resolution_results.xlsx')

    kwargs = {'min_sigma' : min_sigma,
    'max_sigma' : max_sigma,
    'threshold' : threshold,
    'overlap' : overlap,
    'subset_size' : subset_size,
    'pixel_size' : pixel_size,
    'bounds' : bounds,
    'bands' : bands,
    'min_thr' : min_thr,
    'transition_low_limit' : transition_low_limit,
    'transition_high_limit' : transition_high_limit,
    'verbose' : verbose,
    'disp_res' : disp_res,
    'title' : ' ',
    'nbins' : nbins,
    'save_data_xlsx' : save_data,
    'results_file_xlsx' : results_file_xlsx}

    results_file_xlsx, blobs_LoG, error_flags, tr_results, hst_datas =  select_blobs_LoG_analyze_transitions(frame_eval, **kwargs)
    if save_data and disp_res:
        res = plot_blob_map_and_results_single_image(frame_eval, results_file_xlsx, save_png=True)
        res = plot_blob_examples_single_image(frame_eval, results_file_xlsx, save_png=True)

    tr_results_arr = np.array(tr_results)
    frame_ind_arr = np.full(len(error_flags), frame_ind, dtype=int)
    Xpt1 = tr_results_arr[error_flags==0, 1]
    Xpt2 = tr_results_arr[error_flags==0, 2]
    Ypt1 = tr_results_arr[error_flags==0, 3]
    Ypt2 = tr_results_arr[error_flags==0, 4]
    XY_transitions = np.array([Xpt1, Xpt2, Ypt1, Ypt2]).T
    return  frame_ind_arr, error_flags, blobs_LoG, tr_results_arr


class FIBSEM_dataset: 
    """
    A class representing a FIB-SEM data set
    ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
    Contains the info/settings on the FIB-SEM dataset and the procedures that can be performed on it.

    Attributes
    ----------
    fls : array of str
        filenames for the individual data frames in the set
    data_dir : str
        data directory (path)
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
    targ_vector = 1D float array
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
        Note that the its meaning is different from the contrastThreshold,
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
    int_order : int
        The order of interpolation (when transforming the data).
            The order has to be in the range 0-5:
                0: Nearest-neighbor
                1: Bi-linear (default)
                2: Bi-quadratic
                3: Bi-cubic
                4: Bi-quartic
                5: Bi-quintic
    subtract_linear_fit : [boolean, boolean]
        List of two Boolean values for two directions: X- and Y-.
        If True, the linear slopes along X- and Y- directions (respectively)
        will be subtracted from the cumulative shifts.
        This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
    pad_edges : boolean
        If True, the data will be padded before transformation to avoid clipping.
    perform_deformation : boolean
        If True - the data is deformed (in addition to tyransformation defined above) using the deformation field data defined below
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

    Methods
    -------
    SIFT_evaluation(eval_fls = [], **kwargs):
        Evaluate SIFT settings and perfromance of few test frames (eval_fls).

    convert_raw_data_to_tif_files(**kwargs):
        Convert binary ".dat" files into ".tif" files

    evaluate_FIBSEM_statistics(**kwargs):
        Evaluates parameters of FIBSEM data set (data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).

    extract_keypoints(**kwargs):
        Extract Key-Points and Descriptors

    determine_transformations(**kwargs):
        Determine transformation matrices for sequential frame pairs

    process_transformation_matrix(**kwargs):
        Calculate cumulative transformation matrix

    calculate_residual_deformation_fields(**kwargs):
        Calculates residual deformation fields for transformation IN ADDITION to that determined by transformation_matrix (above)

    save_parameters(**kwargs):
        Save transformation attributes and parameters (including transformation matrices)

    check_for_nomatch_frames(thr_npt, **kwargs):
        Check for frames with low number of Key-Point matches,m exclude them and re-calculate the cumulative transformation matrix

    transform_and_save(**kwargs):
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc file

    show_eval_box(**kwargs):
        Show the box used for evaluating the registration quality

    estimate_SNRs(**kwargs):
        Estimate SNRs in Image A and Image B based on single-image SNR calculation.

    evaluate_ImgB_fractions(ImgB_fractions, frame_inds, **kwargs):
        Calculate NCC and SNR vs Image B fraction over a set of frames.

    estimate_resolution_blobs_2D(**kwargs)
        Estimate transitions in the image, uses select_blobs_LoG_analyze_transitions(frame_eval, **kwargs).
    """

    def __init__(self, fls, **kwargs):
        """
        Initializes (or recalls) an instance of  FIBSEM_dataset object. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com

        Parameters:
        ----------
        fls : array of str
            Filenames for the individual data frames in the set

        kwargs:
        ---------
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif).
        recall_parameters : boolean
            If True and dump_filename kwarg points to a valid binary file, will recall the dataset saved into that dump_filename. Default is False.
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
            Choose from the following options (default is RegularizedAffineTransform):
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
        zbin_factor : int
            Binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
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
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params above.
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-. Default is [True, True].
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        subtract_FOVtrend_from_fit : [boolean, boolean]
            If True, FOV trends (image shifts performed during imaging) will be subtracted first, so they do not bias the linear trends in the line above.
            Default is [True, True].
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
        eval_metrics : list of strings
            List of metrics used for evaluation of registration (and resolution). Options are; 'NSAD', 'NCC', 'NMI', 'FSC'. Default is ['NSAD', 'NCC', 'NMI'].
        disp_res : boolean
            If False, the intermediate printouts will be suppressed. Default is True.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is True.
        """

        disp_res = kwargs.get('disp_res', True)
        self.fls = fls
        self.fnms = [os.path.splitext(fl)[0] + '_kpdes.bin' for fl in fls]
        self.nfrs = len(fls)
        self.data_dir = kwargs.get('data_dir', os.getcwd())
        self.ftype = kwargs.get("ftype", 0) # ftype=0 - Shan Xu's binary format  ftype=1 - tif files
        mid_frame = FIBSEM_frame(fls[self.nfrs//2], ftype = self.ftype, calculate_scaled_images=False)
        self.XResolution = kwargs.get("XResolution", mid_frame.XResolution)
        self.YResolution = kwargs.get("YResolution", mid_frame.YResolution)
        self.XResolutions = kwargs.get('XResolutions', np.full(len(fls), mid_frame.XResolution))
        self.YResolutions = kwargs.get('YResolutions', np.full(len(fls), mid_frame.YResolution))
        self.Scaling = kwargs.get("Scaling", mid_frame.Scaling)
        if hasattr(mid_frame, 'PixelSize'):
            self.PixelSize = kwargs.get("PixelSize", mid_frame.PixelSize)
        else:
            self.PixelSize = kwargs.get("PixelSize", 8.0)
        self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        if hasattr(self, 'YResolution'):
            YResolution_default = self.YResolution
        else:
            YResolution_default = FIBSEM_frame(self.fls[len(self.fls)//2], calculate_scaled_images=False).YResolution
        YResolution = kwargs.get("YResolution", YResolution_default)

        test_frame = FIBSEM_frame(fls[0], ftype=self.ftype, calculate_scaled_images=False)
        self.DetA = test_frame.DetA
        self.DetB = test_frame.DetB
        self.ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        self.Sample_ID = kwargs.get("Sample_ID", '')
        self.EightBit = kwargs.get("EightBit", 1)
        self.DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        self.thr_min = kwargs.get("thr_min", 1e-3)
        self.thr_max = kwargs.get("thr_max", 1e-3)
        self.nbins = kwargs.get("nbins", 256)
        self.sliding_minmax = kwargs.get("sliding_minmax", True)
        self.SIFT_nfeatures = kwargs.get("SIFT_nfeatures", 0)
        self.SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", 3)
        self.SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", 0.04)
        self.SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", 10)
        self.SIFT_sigma = kwargs.get("SIFT_sigma", 1.6)
        self.BFMatcher = kwargs.get("BFMatcher", False)           # If True, the BF Matcher is used for keypont matching, otherwise FLANN will be used
        self.save_matches = kwargs.get("save_matches", True)      # If True, matches will be saved into individual files
        self.TransformType = kwargs.get("TransformType", RegularizedAffineTransform)
        self.tr_matr_cum_residual = [np.eye(3,3) for i in np.arange(self.nfrs)]  # placeholder - identity transformation matrix
        l2_param_default = 1e-5                                  # regularization strength (shrinkage parameter)
        l2_matrix_default = np.eye(6)*l2_param_default             # initially set equal shrinkage on all coefficients
        l2_matrix_default[2,2] = 0                                 # turn OFF the regularization on shifts
        l2_matrix_default[5,5] = 0                                 # turn OFF the regularization on shifts
        self.l2_matrix = kwargs.get("l2_matrix", l2_matrix_default)
        self.targ_vector = kwargs.get("targ_vector", np.array([1, 0, 0, 0, 1, 0]))   # target transformation is shift only: Sxx=Syy=1, Sxy=Syx=0
        self.solver = kwargs.get("solver", 'RANSAC')
        self.RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", 0.005)  # fraction of data points for initial RANSAC iteration step.
        self.Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)
        self.drmax = kwargs.get("drmax", 1.5)
        self.max_iter = kwargs.get("max_iter", 1000)

        self.save_res_png  = kwargs.get("save_res_png", True)
        self.zbin_factor =  kwargs.get("zbin_factor", 1)         # binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
        self.eval_metrics = kwargs.get('eval_metrics', ['NSAD', 'NCC', 'NMI'])
        self.fnm_types = kwargs.get("fnm_types", ['mrc'])
        self.flipY = kwargs.get("flipY", False)                     # If True, the registered data will be flipped along Y axis
        self.preserve_scales =  kwargs.get("preserve_scales", True) # If True, the transformation matrix will be adjusted using teh settings defined by fit_params below
        self.fit_params =  kwargs.get("fit_params", ['SG', len(fls)//100+1, 3])          # perform the above adjustment using  Savitzky-Golay (SG) fith with parameters
                                                                    # window size 701, polynomial order 3
        self.int_order = kwargs.get("int_order", False)             #     The order of interpolation. The order has to be in the range 0-5:
                                                                    #    - 0: Nearest-neighbor
                                                                    #    - 1: Bi-linear (default)
                                                                    #    - 2: Bi-quadratic
                                                                    #    - 3: Bi-cubic
                                                                    #    - 4: Bi-quartic
                                                                    #    - 5: Bi-quintic
        self.subtract_linear_fit =  kwargs.get("subtract_linear_fit", [True, True])   # If True, the linear slope will be subtracted from the cumulative shifts.
        self.subtract_FOVtrend_from_fit = kwargs.get("subtract_FOVtrend_from_fit", [True, True])
        self.FOVtrend_x = np.zeros(len(fls))
        self.FOVtrend_y = np.zeros(len(fls))
        self.pad_edges =  kwargs.get("pad_edges", True)
        self.perform_deformation = kwargs.get("perform_deformation", False)
        self.deformation_type = kwargs.get("deformation_type", 'post_1DY')
        self.deformation_sigma = kwargs.get('deformation_sigma', 50)
        build_fnm_reg, build_dtp = build_filename(fls[0], **kwargs)
        self.fnm_reg = kwargs.get("fnm_reg", build_fnm_reg)
        self.dtp = kwargs.get("dtp", build_dtp)
        kwargs.update({'data_dir' : self.data_dir, 'fnm_reg' : self.fnm_reg, 'dtp' : self.dtp})

        if kwargs.get("recall_parameters", False):
            dump_filename = kwargs.get("dump_filename", '')
            try:
                dump_data = pickle.load(open(dump_filename, 'rb'))
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Loaded the data from the dump filename: ', dump_filename)
                dump_loaded = True
            except Exception as ex1:
                dump_loaded = False
                if disp_res:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Failed to open Parameter dump filename: ', dump_filename)
                    print(ex1.message)
            if dump_loaded:
                try:
                    for key in tqdm(dump_data, desc='Recalling the data set parameters'):
                        setattr(self, key, dump_data[key])
                except Exception as ex2:
                    if disp_res:
                        print('Parameter dump filename: ', dump_filename)
                        print('Failed to restore the object parameters')
                        print(ex2.message)
        else:
            if disp_res:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Registered data will be saved into: ', self.fnm_reg)
        if disp_res:
            print('Total Number of frames: ', len(self.fls))
 

    def SIFT_evaluation(self, eval_fls = [], **kwargs):
        '''
        Evaluate SIFT settings and perfromance of few test frames (eval_fls). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        Parameters:
        ---------
        eval_fls : array of str
            filenames for the data frames to be used for SIFT evaluation.
        
        kwargs:
        ---------
        memory_profiling : boolean
            If True, memory profiling will be performed. Default is False.
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        number_of_repeats : int
            Number of repeats of the calculations (under the same conditions). Default is 1. Used for debugging (consistency and determinism).
        data_dir : str
            Data directory (path). Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for key-point extraction
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
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
        BFMatcher : boolean
            If True, the BF Matcher is used for Key-Point matching, otherwise FLANN will be used. Default is object attribute.
        save_matches : boolean
            If True, matches will be saved into individual files. Default is object attribute.
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
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression.
            Default is object attribute.
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg). Default is object attribute.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is object attribute.
        start : string
            Start of search for determining FWHM of the error distributions. Options are 'edges' (default) or 'center'.
        estimation : string
            Returns a width of interval determined using search direction from above or total number of bins above half max. Options are 'interval' (default) or 'count'.
    
        Returns:
        dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts, error_FWHMx, error_FWHMy
        '''
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)

        number_of_repeats = kwargs.get('number_of_repeats', 1)
        if len(eval_fls) == 0:
            eval_fls = [self.fls[self.nfrs//2], self.fls[self.nfrs//2+1]]
        data_dir = kwargs.get("data_dir", self.data_dir)
        ftype = kwargs.get("ftype", self.ftype)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        TransformType = kwargs.get("TransformType", self.TransformType)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        if hasattr(self, "Lowe_Ratio_Threshold"):
            Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", self.Lowe_Ratio_Threshold)
        else:
            Lowe_Ratio_Threshold = kwargs.get("Lowe_Ratio_Threshold", 0.7)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        start = kwargs.get('start', 'edges')
        estimation = kwargs.get('estimation', 'interval')
        verbose = kwargs.get('verbose', True)
        memory_profiling = kwargs.get('memory_profiling', False)

        SIFT_evaluation_kwargs = {'DASK_client' : DASK_client,
                                'DASK_client_retries' : DASK_client_retries,
                                'use_DASK' : use_DASK,
                                'number_of_repeats' : number_of_repeats,
                                'ftype' : ftype,
                                'Sample_ID' : Sample_ID,
                                'data_dir' : data_dir,
                                'fnm_reg' : fnm_reg,
                                'thr_min' : thr_min,
                                'thr_max' : thr_max,
                                'nbins' : nbins,
                                'evaluation_box' : evaluation_box,
                                'TransformType' : TransformType, 
                                'l2_matrix' : l2_matrix,
                                'targ_vector' : targ_vector,
                                'solver' : solver,
                                'RANSAC_initial_fraction' : RANSAC_initial_fraction,
                                'drmax' : drmax,
                                'max_iter' : max_iter,
                                'SIFT_Transform' : TransformType,
                                'SIFT_nfeatures' : SIFT_nfeatures,
                                'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
                                'SIFT_contrastThreshold' : SIFT_contrastThreshold,
                                'SIFT_edgeThreshold' : SIFT_edgeThreshold,
                                'SIFT_sigma' : SIFT_sigma,
                                'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
                                'BFMatcher' : BFMatcher,
                                'save_matches' : save_matches,
                                'verbose' : verbose,
                                'start' : start,
                                'estimation' : estimation,
                                'memory_profiling' : memory_profiling,
                                'save_res_png'  : save_res_png}
        
        dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts, error_FWHMx, error_FWHMy = SIFT_evaluation_dataset(eval_fls, **SIFT_evaluation_kwargs)
        src_pts_filtered, dst_pts_filtered = kpts
        print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Transformation Matrix determined using '+ TransformType.__name__ +' using ' + solver + ' solver')
        print(transform_matrix)
        print('{:d} keypoint matches were detected with {:.1f} pixel outlier threshold'.format(n_matches, drmax))
        print('Number of iterations: {:d}'.format(iteration))
        return dmin, dmax, comp_time, transform_matrix, n_matches, iteration, kpts, error_FWHMx, error_FWHMy


    def convert_raw_data_to_tif_files(self, **kwargs):
        '''
        Convert binary ".dat" files into ".tif" files. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client.
            If set to empty string '' (default), local computations are performed
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        '''
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        if self.ftype ==0 :
            print('Step 2a: Creating "*InLens.tif" files using DASK distributed')
            t00 = time.time()
            if use_DASK:
                try:
                    futures = DASK_client.map(save_inlens_data, self.fls, retries = DASK_client_retries)
                    fls_new = np.array(DASK_client.gather(futures))
                except:
                    fls_new = []
                    for fl in tqdm(self.fls, desc = 'Converting .dat data files into .tif format'):
                            fls_new.append(save_inlens_data(fl))
            else:
                fls_new = []
                for fl in tqdm(self.fls, desc = 'Converting .dat data files into .tif format'):
                    fls_new.append(save_inlens_data(fl))

            t01 = time.time()
            print('Step 2a: Elapsed time: {:.2f} seconds'.format(t01 - t00))
            print('Step 2a: Quick check if all files were converted: ', np.array_equal(self.fls, fls_new))
        else:
            print('Step 2a: data is already in TIF format')


    def evaluate_FIBSEM_statistics(self, **kwargs):
        '''
        Evaluates parameters of FIBSEM data set (Min/Max, Working Distance (WD), Milling Y Voltage (MV), FOV center positions). ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
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
        sliding_minmax : boolean
            If True - data min and max will be taken from data_min_sliding and data_max_sliding arrays. Default is object attribute.
            If False - same data_min_glob and data_max_glob will be used for all files.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Default is object attribute.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        Mill_Volt_Rate_um_per_V : float
            Milling Voltage to Z conversion (µm/V). Default is 31.235258870176065.
        FIBSEM_Data_xlsx : str
            File path of the Excell file for the FIBSEM data set data to be saved (Data Min/Max, Working Distance, Milling Y Voltage, FOV center positions).
        use_existing_data : boolean
            Default is False. If True and the data exists (saved into XLSX), use that.            
        disp_res : boolean
            If True (default), intermediate messages and results will be displayed.

        Returns:
        list of 14 parameters: FIBSEM_Data_xlsx, data_min_glob, data_max_glob, data_min_sliding, data_max_sliding, mill_rate_WD, mill_rate_MV, center_x, center_y, ScanRate, EHT, SEMSpecimenI, XResolutions, YResolutions
            FIBSEM_Data_xlsx : str
                path to Excel file with the FIBSEM data
            data_min_glob : float   
                min data value for I8 conversion (open CV SIFT requires I8)
            data_man_glob : float   
                max data value for I8 conversion (open CV SIFT requires I8)
            data_min_sliding : float array
                min data values (one per file) for I8 conversion
            data_max_sliding : float array
                max data values (one per file) for I8 conversion
            mill_rate_WD : float array
                Milling rate calculated based on Working Distance (WD)
            mill_rate_MV : float array
                Milling rate calculated based on Milling Y Voltage (MV)
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
        use_DASK, status_update_address = check_DASK(DASK_client)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        frame_inds = kwargs.get("frame_inds", np.arange(len(self.fls)))
        data_dir = self.data_dir
        thr_min = kwargs.get("thr_min", self.thr_min)
        thr_max = kwargs.get("thr_max", self.thr_max)
        nbins = kwargs.get("nbins", self.nbins)
        sliding_minmax = kwargs.get("sliding_minmax", self.sliding_minmax)
        fit_params = kwargs.get("fit_params", self.fit_params)

        if hasattr(self, 'Mill_Volt_Rate_um_per_V'):
            Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", self.Mill_Volt_Rate_um_per_V)
        else:
            Mill_Volt_Rate_um_per_V = kwargs.get("Mill_Volt_Rate_um_per_V", 31.235258870176065)

        FIBSEM_Data_xlsx_default = os.path.join(data_dir, self.fnm_reg.replace('.mrc', '_FIBSEM_Data.xlsx'))
        FIBSEM_Data_xlsx = kwargs.get('FIBSEM_Data_xlsx', FIBSEM_Data_xlsx_default)
        disp_res = kwargs.get('disp_res', True)
        use_existing_data = kwargs.get('use_existing_data', False)

        local_kwargs = {'use_DASK' : use_DASK,
                        'DASK_client_retries' : DASK_client_retries,
                        'ftype' : ftype,
                        'frame_inds' : frame_inds,
                        'data_dir' : data_dir,
                        'thr_min' : thr_min,
                        'thr_max' : thr_max,
                        'nbins' : nbins,
                        'sliding_minmax' : sliding_minmax,
                        'fit_params' : fit_params,
                        'Mill_Volt_Rate_um_per_V' : Mill_Volt_Rate_um_per_V,
                        'FIBSEM_Data_xlsx' : FIBSEM_Data_xlsx,
                        'disp_res' : disp_res,
                        'use_existing_data' : use_existing_data}

        if disp_res:
            print('Evaluating the parameters of FIBSEM data set (data Min/Max, Working Distance, Milling Y Voltage, FOV center positions, Scan Rate, EHT)')
        self.FIBSEM_Data = evaluate_FIBSEM_frames_dataset(self.fls, DASK_client, **local_kwargs)
        self.data_minmax = self.FIBSEM_Data[0:5]
        WD = self.FIBSEM_Data[5]
        MillingYVoltage = self.FIBSEM_Data[6]

        sv_apert = np.min((51, len(self.FIBSEM_Data[7])//8*2+1))
        self.FOVtrend_x = savgol_filter(self.FIBSEM_Data[7]*1.0, sv_apert, 1) - self.FIBSEM_Data[7][0]
        self.FOVtrend_y = savgol_filter(self.FIBSEM_Data[8]*1.0, sv_apert, 1) - self.FIBSEM_Data[8][0]
        try:
            self.XResolutions = self.FIBSEM_Data[12].astype(int)
            self.YResolutions = self.FIBSEM_Data[13].astype(int)
        except:
            self.XResolutions = np.full(len(WD), self.XResolution).astype(int)
            self.YResolutions = np.full(len(WD), self.YResolution).astype(int)

        self.XResolution = np.max(self.XResolutions)
        self.YResolution = np.max(self.YResolutions)

        WD_fit_coef = np.polyfit(frame_inds, WD, 1)
        rate_WD = WD_fit_coef[0]*1.0e6
    
        MV_fit_coef = np.polyfit(frame_inds, MillingYVoltage, 1)
        rate_MV = MV_fit_coef[0]*Mill_Volt_Rate_um_per_V*-1.0e3

        Z_pixel_size_WD = rate_WD
        Z_pixel_size_MV = rate_MV

        if ftype == 0:
            if disp_res:
                if self.zbin_factor > 1:
                    print('Z pixel (after {:d}-x Z-binning) = {:.2f} nm - based on WD data'.format(self.zbin_factor, Z_pixel_size_WD*self.zbin_factor))
                    print('Z pixel (after {:d}-x Z-binning) = {:.2f} nm - based on Milling Voltage data'.format(self.zbin_factor, Z_pixel_size_MV*self.zbin_factor))
                else:
                    print('Z pixel = {:.2f} nm  - based on WD data'.format(Z_pixel_size_WD))
                    print('Z pixel = {:.2f} nm  - based on Milling Voltage data'.format(Z_pixel_size_MV))

            self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  Z_pixel_size_WD), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        else:
            if disp_res:
                print('No milling rate data is available, isotropic voxel size is set to {:.2f} nm'.format(self.PixelSize))
            self.voxel_size = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])

        return self.FIBSEM_Data


    def extract_keypoints(self, **kwargs):
        '''
        Extract Key-Points and Descriptors. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
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
        fnm_reg : str
            filename for the final registered dataset
        thr_min : float
            CDF threshold for determining the minimum data value. Default is object attribute.
        thr_max : float
            CDF threshold for determining the maximum data value. Default is object attribute.
        nbins : int
            Number of histogram bins for building the PDF and CDF. Default is object attribute.
        sliding_minmax : boolean
            If True - data min and max will be taken from data_min_sliding and data_max_sliding arrays. Default is object attribute.
            If False - same data_min_glob and data_max_glob will be used for all files.
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
        use_existing_data : boolean
            Default is False. If True and this had already been performed, use existing results.
    
        Returns:
        fnms : array of str
            Filenames for binary files containing Key-Points and Descriptors for each frame.
        '''
        if len(self.fls) == 0:
            print('Data set not defined, perform initialization first')
            fnms = []
        else:  
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            ftype = kwargs.get("ftype", self.ftype)
            data_dir = self.data_dir
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            thr_min = kwargs.get("thr_min", self.thr_min)
            thr_max = kwargs.get("thr_max", self.thr_max)
            nbins = kwargs.get("nbins", self.nbins)
            sliding_minmax = kwargs.get("sliding_minmax", self.sliding_minmax)
            data_minmax = kwargs.get("data_minmax", self.data_minmax)
            SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
            SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
            SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
            SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
            SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
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
                        'use_existing_data' : use_existing_data}

            if sliding_minmax:
                params_s3 = [[dts3[0], dts3[1], dts3[2], kpt_kwargs] for dts3 in zip(self.fls, data_min_sliding, data_max_sliding)]
            else:
                params_s3 = [[fl, data_min_glob, data_max_glob, kpt_kwargs] for fl in self.fls]        
            if use_DASK:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
                futures_s3 = DASK_client.map(extract_keypoints_descr_files, params_s3, retries = DASK_client_retries)
                fnms = DASK_client.gather(futures_s3)
            else:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using Local Computation')
                fnms = []
                for j, param_s3 in enumerate(tqdm(params_s3, desc='Extracting Key Points and Descriptors: ')):
                    fnms.append(extract_keypoints_descr_files(param_s3))

            self.fnms = fnms
        return fnms


    def determine_transformations(self, **kwargs):
        '''
        Determine transformation matrices for sequential frame pairs. ©G.Shtengel 10/2021 gleb.shtengel@gmail.com
        
        kwargs:
        ---------
        DASK_client : DASK client. If empty string '' (default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
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
    
        Returns:
        results_s4 : array of lists containing the results:
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
        if len(self.fnms) == 0:
            print('No data on individual key-point data files, peform key-point search')
            results_s4 = []
        else:
            DASK_client = kwargs.get('DASK_client', '')
            use_DASK, status_update_address = check_DASK(DASK_client)
            if hasattr(self, "DASK_client_retries"):
                DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
            else:
                DASK_client_retries = kwargs.get("DASK_client_retries", 3)
            ftype = kwargs.get("ftype", self.ftype)
            TransformType = kwargs.get("TransformType", self.TransformType)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            RANSAC_initial_fraction = kwargs.get("RANSAC_initial_fraction", self.RANSAC_initial_fraction)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
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
                            'save_matches' : save_matches,
                            'Lowe_Ratio_Threshold' : Lowe_Ratio_Threshold,
                            'start' : start,
                            'estimation' : estimation,
                            'use_existing_data' : use_existing_data}

            params_s4 = []
            for j, fnm in enumerate(self.fnms[:-1]):
                fname1 = self.fnms[j]
                fname2 = self.fnms[j+1]
                params_s4.append([fname1, fname2, dt_kwargs])
            if use_DASK:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
                futures4 = DASK_client.map(determine_transformations_files, params_s4, retries = DASK_client_retries)
                #determine_transformations_files returns (transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration)
                results_s4 = DASK_client.gather(futures4)
            else:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using Local Computation')
                results_s4 = []
                for param_s4 in tqdm(params_s4, desc = 'Extracting Transformation Parameters: '):
                    results_s4.append(determine_transformations_files(param_s4))
            #determine_transformations_files returns (transform_matrix, fnm_matches, kpts, error_abs_mean, error_FWHMx, error_FWHMy, iteration)
            self.transformation_matrix = np.nan_to_num(np.array([result[0] for result in results_s4]))
            self.fnms_matches = [result[1] for result in results_s4]
            self.npts = np.nan_to_num(np.array([len(result[2][0])  for result in results_s4]))
            self.error_abs_mean = np.nan_to_num(np.array([result[3] for result in results_s4]))
            self.error_FWHMx = [result[4] for result in results_s4]
            self.error_FWHMy = [result[5] for result in results_s4]
            print('Mean Number of Keypoints :', np.mean(self.npts).astype(np.int64))
        return results_s4


    def process_transformation_matrix(self, **kwargs):
        '''
        Calculate cumulative transformation matrix
        
        kwargs:
        ---------
        data_dir : str
            Data directory (path). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        Sample_ID : str
            Sample ID. Default is object attribute.
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
        BFMatcher : boolean
            If True, the BF Matcher is used for Key-Point matching, otherwise FLANN will be used. Default is object attribute.
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression.
            Default is object attribute.
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg). Default is object attribute.
        save_matches : boolean
            If True, matches will be saved into individual files. Default is object attribute.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is object attribute.
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params above. Default is object attribute.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Default is object attribute.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-. Default is object attribute.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        subtract_FOVtrend_from_fit : [boolean, boolean]
            If True, FOV trends (image shifts performed during imaging) will be subtracted first, so they do not bias the linear trends in the line above.
            Default is object attribute.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping. Default is object attribute.

        Returns:
        tr_matr_cum_residual, tr_matr_cum_xlsx_file : list of 2D arrays of float and the filename of the XLSX file with the transf matrix results
            Cumulative transformation matrices
        '''
        if len(self.transformation_matrix) == 0:
            print('No data on individual key-point matches, peform key-point search / matching first')
            self.tr_matr_cum_residual = []
        else:
            data_dir = kwargs.get("data_dir", self.data_dir)
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            TransformType = kwargs.get("TransformType", self.TransformType)
            SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
            SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
            SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
            SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
            SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
            Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
            l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
            targ_vector = kwargs.get("targ_vector", self.targ_vector)
            solver = kwargs.get("solver", self.solver)
            drmax = kwargs.get("drmax", self.drmax)
            max_iter = kwargs.get("max_iter", self.max_iter)
            BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
            save_matches = kwargs.get("save_matches", self.save_matches)
            save_res_png  = kwargs.get("save_res_png", self.save_res_png )
            preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
            fit_params =  kwargs.get("fit_params", self.fit_params)
            subtract_linear_fit =  kwargs.get("subtract_linear_fit", self.subtract_linear_fit)
            subtract_FOVtrend_from_fit =  kwargs.get("subtract_FOVtrend_from_fit", self.subtract_FOVtrend_from_fit)
            pad_edges =  kwargs.get("pad_edges", self.pad_edges)
            verbose = kwargs.get('verbose', False)
            if verbose:
                print('Transformation Matrix Data is present, will perform post-processing')

            TM_kwargs = {'fnm_reg' : fnm_reg,
                            'data_dir' : data_dir,
                            'TransformType' : TransformType,
                            'SIFT_nfeatures' : SIFT_nfeatures,
                            'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
                            'SIFT_contrastThreshold' : SIFT_contrastThreshold,
                            'SIFT_edgeThreshold' : SIFT_edgeThreshold,
                            'SIFT_sigma' : SIFT_sigma,
                            'Sample_ID' : Sample_ID,
                            'l2_matrix' : l2_matrix,
                            'targ_vector': targ_vector, 
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'save_matches' : save_matches,
                            'save_res_png ' : save_res_png ,
                            'preserve_scales' : preserve_scales,
                            'fit_params' : fit_params,
                            'subtract_linear_fit' : subtract_linear_fit,
                            'subtract_FOVtrend_from_fit' : subtract_FOVtrend_from_fit,
                            'pad_edges' : pad_edges,
                            'verbose' : verbose}

            self.tr_matr_cum_residual, self.transf_matrix_xlsx_file = process_transformation_matrix_dataset(self.transformation_matrix,
                                             self.FOVtrend_x,
                                             self.FOVtrend_y,
                                             self.fnms_matches,
                                             self.npts,
                                             self.error_abs_mean,
                                             self.error_FWHMx,
                                             self.error_FWHMy,
                                             **TM_kwargs)
        return self.tr_matr_cum_residual, self.transf_matrix_xlsx_file


    def calculate_residual_deformation_fields(self, **kwargs):
        '''
        Calculates residual deformation fields for transformation IN ADDITION to that determined by transformation_matrix. ©G.Shtengel 01/2025 gleb.shtengel@gmail.com

        kwargs:
        ----------
        tr_matr_cum_residual : transformation matrix
            Default is object attribute.
        image_shape : target image shape
            Default is object attribute.
        fnms_matches : files with point matches
            Default is object attribute.
        deformation_type : str
            Type of Deformation. Default is object attribute. Options are:
                'post_1DY'  - Default. Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
                'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
        deformation_sigma :  list of 1 or two floats.   Gaussian width of smoothing (units of pixels). Used during calculation of the deformation. Default is 50.
        zero_mean : boolean
            If True (Default), mean value of the deformation field is subtracted for each frame.
        data_dir : str
            Data directory (path). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        verbose : boolean
            Display intermediate results. Default is False.

        Returns deformation_fields, deformation_fields_bin_file
        '''
        tr_matr_cum_residual = kwargs.get("tr_matr_cum_residual", self.tr_matr_cum_residual)
        if len(self.tr_matr_cum_residual) == 0:
            print('No data on individual key-point matches, peform key-point search / matching first')
        else:
            image_shape = kwargs.get('image_shape', [self.YResolution, self.XResolution])
            fnms_matches = kwargs.get('fnms_matches', self.fnms_matches)
            deformation_type = kwargs.get("deformation_type", self.deformation_type)
            deformation_sigma = kwargs.get('deformation_sigma', self.deformation_sigma)
            zero_mean = kwargs.get('zero_mean', True)
            data_dir = kwargs.get("data_dir", self.data_dir)
            fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
            verbose = kwargs.get('verbose', False)

            if verbose:
                print('Transformation Matrix Data is present, will perform post-processing')
            DF_kwargs = {'fnm_reg' : fnm_reg,
                        'data_dir' : data_dir,
                        'deformation_type' : deformation_type,
                        'deformation_sigma' : deformation_sigma,
                        'zero_mean' : zero_mean,
                        'verbose' : verbose}

            self.deformation_fields, self.deformation_fields_bin_file = calculate_residual_deformation_fields_dataset(tr_matr_cum_residual, image_shape, fnms_matches, **DF_kwargs)
        
        return self.deformation_fields, self.deformation_fields_bin_file


    def save_parameters(self, **kwargs):
        '''
        Save transformation attributes and parameters (including transformation matrices).

        kwargs:
        -------
        dump_filename : string
            String containing the name of the binary dump for saving all attributes of the current instance of the FIBSEM_dataset object.

        Returns:
        dump_filename : string
        '''
        default_dump_filename = os.path.join(self.data_dir, self.fnm_reg.replace('.mrc', '_params.bin'))
        dump_filename = kwargs.get("dump_filename", default_dump_filename)

        pickle.dump(self.__dict__, open(dump_filename, 'wb'))

        '''
        npts_fnm = dump_filename.replace('_params.bin', '_Npts_Errs_data.csv')
        Tr_matrix_xls_fnm = dump_filename.replace('_params.bin', '_Transform_Matrix_data.csv')
        
        # Save the keypoint statistics into a CSV file
        columns=['Npts', 'Mean Abs Error']
        npdt = pd.DataFrame(np.vstack((self.npts, self.error_abs_mean)).T, columns = columns, index = None)
        npdt.to_csv(npts_fnm, index = None)
        
        # Save the X-Y shift data and keypoint statistics into a CSV file
        columns=['T00 (Sxx)', 'T01 (Sxy)', 'T02 (Tx)',  
                 'T10 (Syx)', 'T11 (Syy)', 'T12 (Ty)', 
                 'T20 (0.0)', 'T21 (0.0)', 'T22 (1.0)']
        tr_mx_dt = pd.DataFrame(self.transformation_matrix.reshape((len(self.transformation_matrix), 9)), columns = columns, index = None)
        tr_mx_dt.to_csv(Tr_matrix_xls_fnm, index = None)
        '''
        return dump_filename

    def check_for_nomatch_frames(self, thr_npt, **kwargs):
        '''
        Check if there are frames with low number of matches. If yes, exclude them and re-calculate the transformation matrices.

        Parameters:
        -----------
        thr_npt : int
            Minimum number of matches. If the pair has less than this - it is reported as "suspicious" and is excluded.
        kwargs
        ---------
        data_dir : str
            Data directory (path). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        Sample_ID : str
            Sample ID. Default is object attribute.
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
        BFMatcher : boolean
            If True, the BF Matcher is used for Key-Point matching, otherwise FLANN will be used. Default is object attribute.
        drmax : float
            In the case of 'RANSAC' - Maximum distance for a data point to be classified as an inlier.
            In the case of 'LinReg' - outlier threshold for iterative regression.
            Default is object attribute.
        max_iter : int
            Max number of iterations in the iterative procedure above (RANSAC or LinReg). Default is object attribute.
        save_matches : boolean
            If True, matches will be saved into individual files. Default is object attribute.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is object attribute.
        preserve_scales : boolean
            If True, the cumulative transformation matrix will be adjusted using the settings defined by fit_params above. Default is object attribute.
        fit_params : list
            Example: ['SG', 501, 3]  - perform the above adjustment using Savitzky-Golay (SG) filter with parameters - window size 501, polynomial order 3.
            Default is object attribute.
            Other options are:
                ['LF'] - use linear fit with forces start points Sxx and Syy = 1 and Sxy and Syx = 0
                ['PF', 2]  - use polynomial fit (in this case of order 2)
        subtract_linear_fit : [boolean, boolean]
            List of two Boolean values for two directions: X- and Y-. Default is object attribute.
            If True, the linear slopes along X- and Y- directions (respectively)
            will be subtracted from the cumulative shifts.
            This is performed after the optimal frame-to-frame shifts are recalculated for preserve_scales = True.
        subtract_FOVtrend_from_fit : [boolean, boolean]
            If True, FOV trends (image shifts performed during imaging) will be subtracted first, so they do not bias the linear trends in the line above.
            Default is object attribute.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping. Default is object attribute.

        Returns:
        tr_matr_cum_residual : list of 2D arrays of float
            Cumulative transformation matrices
        '''
        self.thr_npt = thr_npt
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        TransformType = kwargs.get("TransformType", self.TransformType)
        SIFT_nfeatures = kwargs.get("SIFT_nfeatures", self.SIFT_nfeatures)
        SIFT_nOctaveLayers = kwargs.get("SIFT_nOctaveLayers", self.SIFT_nOctaveLayers)
        SIFT_contrastThreshold = kwargs.get("SIFT_contrastThreshold", self.SIFT_contrastThreshold)
        SIFT_edgeThreshold = kwargs.get("SIFT_edgeThreshold", self.SIFT_edgeThreshold)
        SIFT_sigma = kwargs.get("SIFT_sigma", self.SIFT_sigma)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        l2_matrix = kwargs.get("l2_matrix", self.l2_matrix)
        targ_vector = kwargs.get("targ_vector", self.targ_vector)
        solver = kwargs.get("solver", self.solver)
        drmax = kwargs.get("drmax", self.drmax)
        max_iter = kwargs.get("max_iter", self.max_iter)
        BFMatcher = kwargs.get("BFMatcher", self.BFMatcher)
        save_matches = kwargs.get("save_matches", self.save_matches)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
        fit_params =  kwargs.get("fit_params", self.fit_params)
        subtract_linear_fit =  kwargs.get("subtract_linear_fit", self.subtract_linear_fit)
        subtract_FOVtrend_from_fit =  kwargs.get("subtract_FOVtrend_from_fit", self.subtract_FOVtrend_from_fit)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
        verbose = kwargs.get('verbose', False)
  
        res_nomatch_check = check_for_nomatch_frames_dataset(self.fls, self.fnms, self.fnms_matches,
                                     self.transformation_matrix, self.error_abs_mean, self.npts,
                                     self.FOVtrend_x, self.FOVtrend_y, self.FIBSEM_Data,
                                     thr_npt,
                                     data_dir = self.data_dir, fnm_reg = self.fnm_reg, ftype=self.ftype)
        frames_to_remove, self.fls, self.fnms, self.fnms_matches, self.error_abs_mean, self.npts, self.transformation_matrix, self.FOVtrend_x, self.FOVtrend_y, self.FIBSEM_Data = res_nomatch_check

        if len(frames_to_remove) > 0:
            TM_kwargs = {'fnm_reg' : fnm_reg,
                            'data_dir' : data_dir,
                            'TransformType' : TransformType,
                            'SIFT_nfeatures' : SIFT_nfeatures,
                            'SIFT_nOctaveLayers' : SIFT_nOctaveLayers,
                            'SIFT_contrastThreshold' : SIFT_contrastThreshold,
                            'SIFT_edgeThreshold' : SIFT_edgeThreshold,
                            'SIFT_sigma' : SIFT_sigma,
                            'Sample_ID' : Sample_ID,
                            'l2_matrix' : l2_matrix,
                            'targ_vector': targ_vector, 
                            'solver' : solver,
                            'drmax' : drmax,
                            'max_iter' : max_iter,
                            'BFMatcher' : BFMatcher,
                            'save_matches' : save_matches,
                            'save_res_png ' : save_res_png ,
                            'preserve_scales' : preserve_scales,
                            'fit_params' : fit_params,
                            'subtract_linear_fit' : subtract_linear_fit,
                            'subtract_FOVtrend_from_fit' : subtract_FOVtrend_from_fit,
                            'pad_edges' : pad_edges,
                            'verbose' : verbose}
            if verbose:
                print('Transformation Matrix Data is present, will perform post-processing')
            self.tr_matr_cum_residual, self.transf_matrix_xlsx_file = process_transformation_matrix_dataset(self.transformation_matrix,
                                             self.FOVtrend_x,
                                             self.FOVtrend_y,
                                             self.fnms_matches,
                                             self.npts,
                                             self.error_abs_mean,
                                             self.error_FWHMx,
                                             self.error_FWHMy,
                                             **TM_kwargs)

        return self.tr_matr_cum_residual, self.transf_matrix_xlsx_file


    def transform_and_save(self, **kwargs):
        '''
        Transform the frames using the cumulative transformation matrix and save the data set into .mrc and/or .h5 file
        
        kwargs:
        ---------
        DASK_client : If empty string ' ' (Default), local computations are performed.
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is object attribute.
        save_transformed_dataset : boolean
            If True (default), the transformed data set will be saved into MRC  and/or .h5 file.
        save_registration_summary : boolean
            If True (default), the registration analysis data will be saved into XLSX file.
        frame_inds : int array (or list)
            Array of frame indices. Default is all frames (to be transformed).
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        data_dir : str
            Data directory (path). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        fnm_types : list of strings
            File type(s) for output data. Options are: ['h5', 'mrc'].
            Defauls is 'mrc'. 'h5' is BigDataViewer HDF5 format, uses npy2bdv package. Use empty list if do not want to save the data.
        ImgB_fraction : float
            Fractional ratio of Image B to be used for constructing the fused image:
            ImageFused = ImageA * (1.0-ImgB_fraction) + ImageB * ImgB_fraction
        add_offset : boolean
            If True - the Dark Count offset will be added before saving to make values positive (set True if saving into BigDataViewer HDF5 - it uses UI16 data format).
        offset : float
            Offset to be subtracted (see above). Default is Scaling[1, 0]*(1.0-ImgB_fraction) + Scaling[1, 1]*ImgB_fraction.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check
        perform_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
        XResolutions : array
            X-sizes of the frames
        YResolutions : array
            Y-sizes of the frames
        pad_edges : boolean
            If True, the frame will be padded to account for frame position and/or size changes.
        invert_data : boolean
            If True - the data is inverted.
        perform_deformation : boolean
            If True - the data is deformed (in addition to tyransformation defined above) using the deformation field data defined below.
        deformation_type : str
            Type of Deformation. Default is object attribute. Options are:
                'post_1DY'  - Default. Deformation is performed AFTER the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'prior_1DY' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only Y-coordinate components (all pixels along X-axis are deformed the same way).
                'post_1DX'  - Deformation is performed AFTER the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'prior_1DX' - Deformation is performed PRIOR to the matrix transformation using 1D deformation field with only X-coordinate components (all pixels along Y-axis are deformed the same way).
                'post_2D'   - Deformation is performed AFTER the matrix transformation using 2D deformation field.
                'prior_2D'  - Deformation is performed PRIOR to the matrix transformation using 2D deformation field.
        deformation_fields : float array
            Deformation Field Array (for deformation set above).
        flatten_image : boolean
            Perform image flattening. Default is object attribute.
        image_correction_file : str
            Full path to a binary filename that contains source name (image_correction_source) and correction array (img_correction_array).
        image_scales : array of floats
            Image multipliers for image rescaling: I = (I-image_offset)*image_scale + image_offset.
        image_offsets : array of floats
            Image offsets for image rescaling: I = (I-image_offset)*image_scale + image_offset.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is object attribute.
        zbin_factor : int
            Binning factor along Z-axis. Default is object attribute.
        int_order : int
            The order of interpolation (when transforming the data). Default is object attribute.
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        dtp : Data Type
            Python data type for saving. Default is np.int16.
        eval_metrics : list of str
            List of evaluation metrics to use. Default is object attribute.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            If evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
        start_evaluation_box : list of 4 int
            see above.
        stop_evaluation_box : list of 4 int
            see above.
        save_sample_frames_png : boolean
            If True, sample frames with superimposed eval box and registration analysis data will be saved into png files.
        dtp  : dtype
            Python data type for saving. Default is int16, the other option currently is np.uint8.
        fill_value : float
            Fill value for padding. Default is zero.
        remove_intermediate_frames : boolean
            If True (default), intermediate frames (TIFF files) will be removed
        disp_res : boolean
            If True (default), intermediate messages and results will be displayed.
        chunked_mrc_write : boolean
            if True, the MRC stack is written in chunks, otherwise (False, Default) frame-by-frame.
        chunk_length : int
            length of MRC write chunk. Default is 32.
        
        Returns:
        reg_summary, reg_summary_xlsx
            reg_summary : pandas DataFrame
            reg_summary = pd.DataFrame(np.vstack((npts, error_abs_mean, image_nsad, image_ncc, image_mi)
            reg_summary_xlsx : name of the XLSX spreadsheet file containing the data
        '''

        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client)

        save_transformed_dataset = kwargs.get('save_transformed_dataset', True)
        save_registration_summary = kwargs.get('save_registration_summary', True)
        frame_inds = kwargs.get('frame_inds', np.arange(len(self.fls)))
    
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        if hasattr(self, 'XResolutions'):
            XResolutions_default = self.XResolutions
        else:
            XResolutions_default = np.full(len(frame_inds), FIBSEM_frame(self.fls[len(self.fls)//2], calculate_scaled_images=False).XResolution)
        XResolutions = kwargs.get("XResolutions", XResolutions_default)
        if hasattr(self, 'YResolution'):
            YResolutions_default = self.YResolutions
        else:
            YResolutions_default = np.full(len(frame_inds), FIBSEM_frame(self.fls[len(self.fls)//2], calculate_scaled_images=False).YResolution)
        YResolutions = kwargs.get("YResolutions", YResolutions_default)

        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        if hasattr(self, 'fnm_types'):
            fnm_types = kwargs.get("fnm_types", self.fnm_types)
        else:
            fnm_types = kwargs.get("fnm_types", ['mrc'])
        if hasattr(self, 'ImgB_fraction'):
            ImgB_fraction = kwargs.get("ImgB_fraction", self.ImgB_fraction)
        else:
            ImgB_fraction = kwargs.get("ImgB_fraction", 0.0)
        if self.DetB == 'None':
            ImgB_fraction = 0.0
        if hasattr(self, 'add_offset'):
            add_offset = kwargs.get("add_offset", self.add_offset)
        else:
            add_offset = kwargs.get("add_offset", False)
        if add_offset:
            offset = kwargs.get('offset', self.Scaling[1, 0] * (1.0-ImgB_fraction) + self.Scaling[1, 1] * ImgB_fraction)
        else:
            offset = 0.0
        save_sample_frames_png = kwargs.get("save_sample_frames_png", True)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        if hasattr(self, 'eval_metrics'):
            eval_metrics =  kwargs.get("eval_metrics", self.eval_metrics)
        else:
            eval_metrics = kwargs.get('eval_metrics', ['NSAD', 'NCC', 'NMI', 'FSC'])
        if hasattr(self, 'zbin_factor'):
            zbin_factor =  kwargs.get("zbin_factor", self.zbin_factor)         # binning factor in z-direction (milling direction). Data will be binned when saving the final result. Default is 1.
        else:
            zbin_factor =  kwargs.get("zbin_factor", 1)
        if hasattr(self, 'voxel_size'):
            voxel_size = kwargs.get("voxel_size", self.voxel_size)
        else:
            voxel_size_default = np.rec.array((self.PixelSize,  self.PixelSize,  self.PixelSize), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
            voxel_size = kwargs.get("voxel_size", voxel_size_default)
        voxel_size_zbinned = voxel_size.copy()
        voxel_size_zbinned.z = voxel_size.z * zbin_factor
        if hasattr(self, 'flipY'):
            flipY = kwargs.get("flipY", self.flipY)
        else:
            flipY = kwargs.get("flipY", False)
        if hasattr(self, 'dump_filename'):
            dump_filename = kwargs.get("dump_filename", self.dump_filename)
        else:
            dump_filename = kwargs.get("dump_filename", '')
        int_order = kwargs.get("int_order", self.int_order) 
        preserve_scales =  kwargs.get("preserve_scales", self.preserve_scales)
        if hasattr(self, 'flatten_image'):
            flatten_image = kwargs.get("flatten_image", self.flatten_image)
        else:
            flatten_image = kwargs.get("flatten_image", False)
        if hasattr(self, 'image_correction_file'):
            image_correction_file = kwargs.get("image_correction_file", self.image_correction_file)
        else:
            image_correction_file = kwargs.get("image_correction_file", '')
        image_scales = kwargs.get("image_scales", np.full(len(self.fls), 1.0))
        image_offsets = kwargs.get("image_offsets", np.zeros(len(self.fls)))
        perform_transformation =  kwargs.get("perform_transformation", True)  and hasattr(self, 'tr_matr_cum_residual')
        if hasattr(self, 'perform_deformation'):
            perform_deformation = kwargs.get("perform_deformation", self.perform_deformation)
        else:
            perform_deformation = kwargs.get("perform_deformation", False)
        if hasattr(self, 'deformation_type'):
            deformation_type = kwargs.get("deformation_type", self.deformation_type)
        else:
            deformation_type = kwargs.get("deformation_type", 'post_1DY')
        if hasattr(self, 'deformation_fields'):
            deformation_fields = kwargs.get("deformation_fields", self.deformation_fields)
        else:
            deformation_fields = kwargs.get("deformation_fields", np.zeros((len(self.fls), self.YResolution), dtype=float))
        if hasattr(self, 'pad_edges'):
            pad_edges = kwargs.get("pad_edges", self.pad_edges)
        else:
            pad_edges = kwargs.get("pad_edges", True)
        if hasattr(self, 'invert_data'):
            invert_data = kwargs.get("invert_data", self.invert_data)
        else:
            invert_data = kwargs.get("invert_data", False)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        disp_res  = kwargs.get("disp_res", True )
        dtp = kwargs.get("dtp", np.int16)  # Python data type for saving. Default is int16, the other option currently is np.uint8.
        fill_value = kwargs.get('fill_value', 0.0) + offset
        remove_intermediate_frames = kwargs.get('remove_intermediate_frames', True)
        chunked_mrc_write = kwargs.get('chunked_mrc_write', False)
        chunk_length = kwargs.get('chunk_length', 32)
        
        save_kwargs = {'fnm_types' : fnm_types,
                        'fnm_reg' : fnm_reg,
                        'use_DASK' : use_DASK,
                        'DASK_client_retries' : DASK_client_retries,
                        'ftype' : ftype,
                        'XResolutions' : XResolutions,
                        'YResolutions' : YResolutions,
                        'data_dir' : data_dir,
                        'voxel_size' : voxel_size_zbinned,
                        'pad_edges' : pad_edges,
                        'ImgB_fraction' : ImgB_fraction,
                        'save_res_png ' : save_res_png ,
                        'dump_filename' : dump_filename,
                        'dtp' : dtp,
                        'fill_value' : fill_value,
                        'zbin_factor' : zbin_factor,
                        'eval_metrics' : eval_metrics,
                        'flipY' : flipY,
                        'int_order' : int_order,                        
                        'preserve_scales' : preserve_scales,
                        'flatten_image' : flatten_image,
                        'image_correction_file' : image_correction_file,
                        'image_scales' : image_scales,
                        'image_offsets' : image_offsets,
                        'perform_transformation' : perform_transformation,
                        'perform_deformation' : perform_deformation,
                        'deformation_type' : deformation_type,
                        'deformation_fields' : deformation_fields,
                        'invert_data' : invert_data,
                        'evaluation_box' : evaluation_box,
                        'sliding_evaluation_box' : sliding_evaluation_box,
                        'start_evaluation_box' : start_evaluation_box,
                        'stop_evaluation_box' : stop_evaluation_box,
                        'save_sample_frames_png' : save_sample_frames_png,
                        'save_registration_summary' : save_registration_summary,
                        'chunked_mrc_write' : chunked_mrc_write,
                        'chunk_length' : chunk_length,
                        'disp_res' : disp_res}

        # first, transform, bin and save frame chunks into individual tif files
        if disp_res:
            print('Transforming and Saving Intermediate Registered Frames')
        end_frame = ((frame_inds[0]+len(frame_inds)-1)//zbin_factor+1)*zbin_factor
        st_frames = np.arange(frame_inds[0], end_frame, zbin_factor)
        registered_filenames = transform_and_save_frames(DASK_client, frame_inds, self.fls, self.tr_matr_cum_residual, **save_kwargs)
        frame0 = tiff.imread(registered_filenames[0])
        ny, nx = np.shape(frame0)
        if disp_res:
            print('Analyzing Registration Quality')
        shape = [self.YResolution, self.XResolution]
        shapes = [self.YResolutions, self.XResolutions]
        if pad_edges and perform_transformation:
            xi, yi, xsz, ysz = determine_sizes_and_offsets(shapes, self.tr_matr_cum_residual)
            #xi, yi, padx, pady = determine_pad_offsets(shape, self.tr_matr_cum_residual)
            # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
            # so that the transformed images are not clipped.
            # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
            # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
            # those are calculated below base on the amount of padding calculated above
            shift_matrix = np.array([[1.0, 0.0, xi],
                                     [0.0, 1.0, yi],
                                     [0.0, 0.0, 1.0]]).astype(float)
            inv_shift_matrix = np.linalg.inv(shift_matrix)
        else:
            xi = 0
            yi = 0
            xsz = shape[1]
            ysz = shape[0]
            shift_matrix = np.eye(3,3)
            inv_shift_matrix = np.eye(3,3)

        local_kwargs = {'start_evaluation_box' : start_evaluation_box,
                         'stop_evaluation_box' : stop_evaluation_box,
                         'sliding_evaluation_box' : sliding_evaluation_box,
                         'pad_edges' : pad_edges,
                         'perform_transformation' : perform_transformation,
                         'tr_matr' : self.tr_matr_cum_residual}
        eval_bounds_all = set_eval_bounds(shape, evaluation_box, **local_kwargs)
        eval_bounds = np.array(eval_bounds_all)[st_frames]
        if flipY:
            xi_evals = np.array(eval_bounds)[:, 0]
            xa_evals = np.array(eval_bounds)[:, 1]
            ya_evals = ysz - np.array(eval_bounds)[:, 2]
            yi_evals = ysz - np.array(eval_bounds)[:, 3]
            eval_bounds = np.vstack((xi_evals, xa_evals, yi_evals, ya_evals)).T
        
        save_kwargs['eval_bounds'] = eval_bounds

        reg_summary, reg_summary_xlsx = analyze_registration_frames(DASK_client, registered_filenames, **save_kwargs)

        if save_transformed_dataset:
            if disp_res:
                print("Creating Dask Array Stack")
            # now build dask array of the transformed dataset
            # read the first file to get the shape and dtype (ASSUMING THAT ALL FILES SHARE THE SAME SHAPE/TYPE)
            lazy_imread = dask.delayed(tiff.imread)  # lazy reader
            lazy_arrays = [lazy_imread(fn) for fn in registered_filenames]
            dask_arrays = [ da.from_delayed(delayed_reader, shape=frame0.shape, dtype=frame0.dtype)   for delayed_reader in lazy_arrays]
            # Stack infividual frames into one large dask.array
            if add_offset:
                FIBSEMstack = da.stack(dask_arrays, axis=0) - offset
            else:
                FIBSEMstack = da.stack(dask_arrays, axis=0)
            #nz, ny, nx = FIBSEMstack.shape
            fnms_saved = save_data_stack(FIBSEMstack, **save_kwargs)
        else:
            if disp_res:
                print('Registered data set is NOT saved into a file')

        if remove_intermediate_frames:
            # Remove Intermediate Registered Frame Files
            for registered_filename in tqdm(registered_filenames, desc='Removing Intermediate Registered Frame Files: ', display = disp_res):
                try:
                    os.remove(registered_filename)
                except:
                    pass

        return reg_summary, reg_summary_xlsx


    def show_eval_box(self, **kwargs):
        '''
        Show the box used for evaluating the registration quality.

        kwargs:
        ---------
        frame_inds : array or list of int
            Array or list of frame indices to use to display the evaluation box.
            Default is [nfrs//10, nfrs//2, nfrs//10*9].
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            If evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
        start_evaluation_box : list of 4 int
            see above.
        stop_evaluation_box : list of 4 int
            see above.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        data_dir : str
            Data directory (path). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        Sample_ID : str
            Sample ID. Default is object attribute.
        int_order : int
            The order of interpolation (when transforming the data). Default is object attribute.
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        perform_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed.
        invert_data : boolean
            If True - the data is inverted.
        box_linewidth : float
            Linewidth for the box outline. Default is 1.0.
        box_color : color
            Color for the box outline. Default is 'yellow'.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check.
        pad_edges : boolean
            If True, the data will be padded before transformation to avoid clipping.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is False.
        fill_value : float
            Fill value for padding. Default is 0.
        verbose : boolean
            Display intermediate comments / results. Default is False.
        
        '''
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        int_order = kwargs.get("int_order", self.int_order) 
        perform_transformation =  kwargs.get("perform_transformation", True) and hasattr(self, 'tr_matr_cum_residual')
        #print('perform_transformation: ', perform_transformation)
        invert_data =  kwargs.get("invert_data", False)
        box_linewidth = kwargs.get("box_linewidth", 1.0)
        box_color = kwargs.get("box_color", 'yellow')
        flipY = kwargs.get("flipY", False)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
        fill_value = kwargs.get('fill_value', 0.0)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        fls = self.fls
        nfrs = len(fls)
        default_indices = [nfrs//10, nfrs//2, nfrs//10*9]
        frame_inds = kwargs.get("frame_inds", default_indices)
        verbose = kwargs.get("verbose", False)
        shape = [self.YResolution, self.XResolution]
        shapes = [self.YResolutions, self.XResolutions] 
        if pad_edges and perform_transformation:
            xi, yi, xsz, ysz = determine_sizes_and_offsets(shapes, self.tr_matr_cum_residual)
            # xi, yi, padx, pady = determine_pad_offsets(shape, self.tr_matr_cum_residual)
            # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
            # so that the transformed images are not clipped.
            # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
            # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
            # those are calculated below base on the amount of padding calculated above
            shift_matrix = np.array([[1.0, 0.0, xi],
                                     [0.0, 1.0, yi],
                                     [0.0, 0.0, 1.0]]).astype(float)
            inv_shift_matrix = np.linalg.inv(shift_matrix)
        else:
            xi = 0
            yi = 0
            xsz = shape[1]
            ysz = shape[0]
            shift_matrix = np.eye(3,3)
            inv_shift_matrix = np.eye(3,3)
        if verbose:
            print('Results of determine_sizes_and_offsets:')
            print('xi, yi, xsz, ysz: ', xi, yi, xsz, ysz)

        local_kwargs = {'start_evaluation_box' : start_evaluation_box,
                         'stop_evaluation_box' : stop_evaluation_box,
                         'sliding_evaluation_box' : sliding_evaluation_box,
                         'pad_edges' : pad_edges,
                         'perform_transformation' : perform_transformation,
                         'tr_matr' : self.tr_matr_cum_residual,
                         'frame_inds' : frame_inds}
        eval_bounds = set_eval_bounds(shape, evaluation_box, **local_kwargs)
        '''
        #print('PADS: ', padx, pady)
        #print('Sizes: ', xsz, ysz)
        #print('eval_bounds: ', eval_bounds)

        local_kwargs = {'start_evaluation_box' : start_evaluation_box,
                         'stop_evaluation_box' : stop_evaluation_box,
                         'sliding_evaluation_box' : sliding_evaluation_box,
                         'pad_edges' : pad_edges,
                         'perform_transformation' : perform_transformation,
                         'tr_matr' : self.tr_matr_cum_residual}
        eval_bounds_all = set_eval_bounds(shape, evaluation_box, **local_kwargs)
        #print('eval_bounds_check: ', np.array(eval_bounds_all)[frame_inds])
        eval_bounds = np.array(eval_bounds_all)[frame_inds]
        '''
        if verbose:
            print('Will analyze frames with inds:', frame_inds)
            print('Frame files:', np.array(fls)[frame_inds])
            print('Will use fill_value = {:.3f} for padding'.format(fill_value))
            if perform_transformation:
                print('Will perform transformation')
            else:
                print('Will NOT perform transformation')
        for j,fr_ind in enumerate(frame_inds):
            frame = FIBSEM_frame(fls[fr_ind], ftype=ftype, calculate_scaled_images=False)
            #frame_img = np.zeros((ysz, xsz), dtype=float) + fill_value
            frame_img = np.full((ysz, xsz), fill_value, dtype=float)
            frame_img[yi:yi+frame.YResolution, xi:xi+frame.XResolution]  = frame.RawImageA.astype(float)

            if perform_transformation:
                transf = ProjectiveTransform(matrix = shift_matrix @ (self.tr_matr_cum_residual[fr_ind] @ inv_shift_matrix))
                #print('Frame: ', fr_ind, ',  transformation matrix: ', self.tr_matr_cum_residual[fr_ind])
                #print('shift matrices', shift_matrix, inv_shift_matrix)
                #print('direct matrix: ', shift_matrix @ (self.tr_matr_cum_residual[fr_ind] @ inv_shift_matrix))
                #print('inverse matrix: ', np.linalg.inv(shift_matrix @ (self.tr_matr_cum_residual[fr_ind] @ inv_shift_matrix)))
                frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True, mode='constant', cval=fill_value)
            else:
                frame_img_reg = frame_img.copy()

            if flipY:
                frame_img_reg = np.flip(frame_img_reg, axis=0)
                xi_eval, xa_eval = eval_bounds[j, 0:2]
                ya_eval, yi_eval = ysz - np.array(eval_bounds)[j, 2:4]
            else:
                xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds[j, :]
            #print(xi_eval, xa_eval, yi_eval, ya_eval)
            vmin, vmax = get_min_max_thresholds(frame_img_reg[yi_eval:ya_eval, xi_eval:xa_eval], disp_res=False)
            fig, ax = plt.subplots(1,1, figsize=(10.0, 11.0*ysz/xsz))
            if invert_data:
                cmap = 'Greys_r'
            else:
                cmap = 'Greys'
            ax.imshow(frame_img_reg, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.grid(True, color = "cyan")
            ax.set_title(fls[fr_ind])
            rect_patch = patches.Rectangle((xi_eval,yi_eval), np.abs(xa_eval-xi_eval), np.abs(ya_eval-yi_eval),
                linewidth=box_linewidth, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect_patch)
            if save_res_png :
                fig.savefig(os.path.splitext(fls[fr_ind])[0]+'_evaluation_box.png', dpi=300)


    def estimate_SNRs(self, **kwargs):
        '''
        Estimate SNRs in Image A and Image B based on single-image SNR calculation using Auto-Correlation.  

        kwargs:
        ---------
        frame_inds : list of int
            Array or list of frame indices to use to display the evaluation box. Default is [nfrs//10, nfrs//2, nfrs//10*9].
        zero_mean: boolean
            If True (default), auto-correlation is zero-mean. Default is True.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            If evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
        start_evaluation_box : list of 4 int
            see above.
        stop_evaluation_box : list of 4 int
            see above.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        data_dir : str
            Data directory (path). Default is object attribute.
        fnm_reg : str
            Filename for the final registered dataset. Default is object attribute.
        Sample_ID : str
            Sample ID. Default is object attribute.
        ImgB_fraction : float
            Optional fractional weight of Image B to use for constructing the fused image: FusedImage = ImageA*(1.0-ImgB_fraction) + ImageB*ImgB_fraction
            If not provided, the value determined from rSNR ratios will be used.
        perform_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed. Default is False.
        int_order : int
            The order of interpolation (when transforming the data). Default is object attribute.
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        invert_data : boolean
            If True - the data is inverted. Default is object attribute or False.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is object attribute.
        pad_edges : boolean
            If True, the frame will be padded to account for frame position and/or size changes. Default is object attribute.
        extrapolate_signal : str
            Extrapolation method to find signal auto-correlation at 0-point (“Noise-Free Autocorrelation”). Options are:
                'nearest'  - nearest point (1 pixel away from center, same as NN in [1]).
                'linear'   - linear interpolation of 2-points next to center (same as FO in [1]).
                'parabolic' - parabolic interpolation of 2 point left and 2 points right (for 4-point interpolation this is the same as NN+FO in [1]).
                'gaussian'  - gaussian interpolation with number of points = aperture
                'LDR' - use Levinson-Durbin recusrsion (ACLDR in [1]).
            Default is 'parabolic'.
        edge_fraction : float
            Fraction of the full auto-correlation  range used to calculate the "zero value" (default is 0.10).
        nlags : int
            In case of 'LDR' (Levinson-Durbin recursion) nlags is the recursion order (a number of lags). Default is min(xsize/4, ysize/4).
        aperture : int
            Total number of points for gaussian interpolation. Default is 6.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check. Default is False.

        Returns:
            ImgB_fraction_xSNR, ImgB_fraction_ySNR, ImgB_fraction_rSNR
        
        '''
        fls = self.fls
        nfrs = len(fls)
        default_indices = [nfrs//10, nfrs//2, nfrs//10*9]
        frame_inds = kwargs.get("frame_inds", default_indices)
        zero_mean = kwargs.get('zero_mean', True)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        ImgB_fraction = kwargs.get("ImgB_fraction", 0.00 )
        perform_transformation =  kwargs.get("perform_transformation", False) and hasattr(self, 'tr_matr_cum_residual')
        int_order = kwargs.get("int_order", self.int_order)
        if hasattr(self, 'invert_data'):
            invert_data =  kwargs.get("invert_data", self.invert_data)
        else:
            invert_data =  kwargs.get("invert_data", False)
        flipY = kwargs.get("flipY", self.flipY)
        pad_edges =  kwargs.get("pad_edges", self.pad_edges)
        extrapolate_signal = kwargs.get('extrapolate_signal', 'parabolic')
        edge_fraction = kwargs.get("edge_fraction", 0.10)
        shape = [self.YResolution, self.XResolution]
        shapes = [self.YResolutions, self.XResolutions]
        nlags = kwargs.get("nlags", np.min(shape)//4)
        aperture = kwargs.get("aperture", 6)
        save_res_png  = kwargs.get("save_res_png", False )
        if pad_edges and perform_transformation:
            xi, yi, xsz, ysz = determine_sizes_and_offsets(shapes, self.tr_matr_cum_residual)
            # xi, yi, padx, pady = determine_pad_offsets(shape, self.tr_matr_cum_residual)
            # The initial transformation matrices are calculated with no padding.Padding is done prior to transformation
            # so that the transformed images are not clipped.
            # Such padding means shift (by xi and yi values). Therefore the new transformation matrix
            # for padded frames will be (Shift Matrix)x(Transformation Matrix)x(Inverse Shift Matrix)
            # those are calculated below base on the amount of padding calculated above
            shift_matrix = np.array([[1.0, 0.0, xi],
                                     [0.0, 1.0, yi],
                                     [0.0, 0.0, 1.0]]).astype(float)
            inv_shift_matrix = np.linalg.inv(shift_matrix)
        else:
            xi = 0
            yi = 0
            xsz = shape[1]
            ysz = shape[0]
            shift_matrix = np.eye(3,3)
            inv_shift_matrix = np.eye(3,3)

        frame_img = np.zeros((ysz, xsz))
        xSNRAs=[]
        ySNRAs=[]
        rSNRAs=[]
        xSNRBs=[]
        ySNRBs=[]
        rSNRBs=[]

        local_kwargs = {'start_evaluation_box' : start_evaluation_box,
                 'stop_evaluation_box' : stop_evaluation_box,
                 'sliding_evaluation_box' : sliding_evaluation_box,
                 'pad_edges' : pad_edges,
                 'perform_transformation' : perform_transformation,
                 'tr_matr' : self.tr_matr_cum_residual,
                 'frame_inds' : frame_inds}
        eval_bounds = set_eval_bounds(shape, evaluation_box, **local_kwargs)

        for j, frame_ind in enumerate(tqdm(frame_inds, desc='Analyzing Auto-Correlation SNRs ')):
            frame = FIBSEM_frame(fls[frame_ind], ftype=ftype, calculate_scaled_images=False)
            xa = xi+frame.XResolution
            ya = yi+frame.YResolution
            xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds[j, :]

            frame_imgA = np.zeros((ysz, xsz), dtype=float)
            if self.DetB != 'None':
                frame_imgB = np.zeros((ysz, xsz), dtype=float)

            if invert_data:
                frame_imgA[yi:ya, xi:xa] = np.negative(frame.RawImageA.astype(float))
                if self.DetB != 'None':
                    frame_imgB[yi:ya, xi:xa] = np.negative(frame.RawImageB.astype(float))
            else:
                frame_imgA[yi:ya, xi:xa]  = frame.RawImageA.astype(float)
                if self.DetB != 'None':
                    frame_imgB[yi:ya, xi:xa]  = frame.RawImageB.astype(float)

            if perform_transformation:
                transf = ProjectiveTransform(matrix = shift_matrix @ (self.tr_matr_cum_residual[frame_ind] @ inv_shift_matrix))
                frame_imgA_reg = warp(frame_imgA, transf, order = int_order, preserve_range=True)
                if self.DetB != 'None':
                    frame_imgB_reg = warp(frame_imgB, transf, order = int_order, preserve_range=True)
            else:
                frame_imgA_reg = frame_imgA.copy()
                if self.DetB != 'None':
                    frame_imgB_reg = frame_imgB.copy()

            if flipY:
                frame_imgA_reg = np.flip(frame_imgA_reg, axis=0)
                if self.DetB != 'None':
                    frame_imgB_reg = np.flip(frame_imgB_reg, axis=0)
            if zero_mean:
                frame_imgA_eval = frame_imgA_reg[yi_eval:ya_eval, xi_eval:xa_eval]
            else:
                frame_imgA_eval = frame_imgA_reg[yi_eval:ya_eval, xi_eval:xa_eval] - self.Scaling[1,0]
            SNR_png = os.path.splitext(os.path.split(fls[frame_ind])[1])[0] + '.png'
            SNR_png_fname = os.path.join(data_dir, SNR_png)
            print('Analyzing the Detector A Image')
            ImageA_xSNR, ImageA_ySNR, ImageA_rSNR= Single_Image_SNR(frame_imgA_eval,
                                                                    zero_mean = zero_mean,
                                                                    extrapolate_signal=extrapolate_signal,
                                                                    edge_fraction = edge_fraction,
                                                                    aperture = aperture,
                                                                    nlags = nlags,
                                                                    save_res_png=save_res_png,
                                                                    res_fname = SNR_png_fname.replace('.png', '_ImgA_SNR.png'),
                                                                    img_label='Image A, frame={:d}'.format(frame_ind))
            xSNRAs.append(ImageA_xSNR)
            ySNRAs.append(ImageA_ySNR)
            rSNRAs.append(ImageA_rSNR)
            if self.DetB != 'None':
                print('Analyzing the Detector B Image')
                if zero_mean:
                    frame_imgB_eval = frame_imgB_reg[yi_eval:ya_eval, xi_eval:xa_eval]
                else:
                    frame_imgB_eval = frame_imgB_reg[yi_eval:ya_eval, xi_eval:xa_eval] - self.Scaling[1,1]
                ImageB_xSNR, ImageB_ySNR, ImageB_rSNR = Single_Image_SNR(frame_imgB_eval,
                                                                        zero_mean = zero_mean,
                                                                        extrapolate_signal=extrapolate_signal,
                                                                        edge_fraction = edge_fraction,
                                                                        aperture = aperture,
                                                                        nlags = nlags,
                                                                        save_res_png=save_res_png,
                                                                        res_fname = SNR_png_fname.replace('.png', '_ImgB_SNR.png'),
                                                                        img_label='Image B, frame={:d}'.format(frame_ind))
                xSNRBs.append(ImageB_xSNR)
                ySNRBs.append(ImageB_ySNR)
                rSNRBs.append(ImageB_rSNR)

        fig, ax = plt.subplots(1,1, figsize = (6,4))
        ax.plot(frame_inds, xSNRAs, 'r+', label='Image A x-SNR')
        ax.plot(frame_inds, ySNRAs, 'b+', label='Image A y-SNR')
        ax.plot(frame_inds, rSNRAs, 'g+', label='Image A r-SNR')
        if self.DetB != 'None':
            ax.plot(frame_inds, xSNRBs, 'rx', linestyle='dotted', label='Image B x-SNR')
            ax.plot(frame_inds, ySNRBs, 'bx', linestyle='dotted', label='Image B y-SNR')
            ax.plot(frame_inds, rSNRBs, 'gx', linestyle='dotted', label='Image B r-SNR')
            #ImgB_fraction_xSNR = np.mean(np.array(xSNRBs)/(np.array(xSNRAs) + np.array(xSNRBs)))
            #ImgB_fraction_ySNR = np.mean(np.array(ySNRBs)/(np.array(ySNRAs) + np.array(ySNRBs)))
            #ImgB_fraction_rSNR = np.mean(np.array(rSNRBs)/(np.array(rSNRAs) + np.array(rSNRBs)))
            ImgB_fraction_xSNR = np.mean(np.array(xSNRBs)/(np.array(xSNRAs)))
            ImgB_fraction_ySNR = np.mean(np.array(ySNRBs)/(np.array(ySNRAs)))
            ImgB_fraction_rSNR = np.mean(np.array(rSNRBs)/(np.array(rSNRAs)))
            if ImgB_fraction < 1e-9:
                ImgB_fraction = ImgB_fraction_rSNR
            ax.text(0.1, 0.5, 'ImgB fraction (x-SNR) = {:.4f}'.format(ImgB_fraction_xSNR), color='r', transform=ax.transAxes)
            ax.text(0.1, 0.42, 'ImgB fraction (y-SNR) = {:.4f}'.format(ImgB_fraction_ySNR), color='b', transform=ax.transAxes)
            ax.text(0.1, 0.34, 'ImgB fraction (r-SNR) = {:.4f}'.format(ImgB_fraction_rSNR), color='g', transform=ax.transAxes)

            xSNRFs=[]
            ySNRFs=[]
            rSNRFs=[]
            for j, frame_ind in enumerate(tqdm(frame_inds, desc='Re-analyzing Auto-Correlation SNRs for fused image')):
                frame = FIBSEM_frame(fls[frame_ind], ftype=ftype, calculate_scaled_images=False)
                xa = xi+frame.XResolution
                ya = yi+frame.YResolution

                frame_imgA = np.zeros((ysz, xsz), dtype=float)
                if self.DetB != 'None':
                    frame_imgB = np.zeros((ysz, xsz), dtype=float)

                if invert_data:
                    frame_imgA[yi:ya, xi:xa] = np.negative(frame.RawImageA.astype(float))
                    if self.DetB != 'None':
                        frame_imgB[yi:ya, xi:xa] = np.negative(frame.RawImageB.astype(float))
                else:
                    frame_imgA[yi:ya, xi:xa]  = frame.RawImageA.astype(float)
                    if self.DetB != 'None':
                        frame_imgB[yi:ya, xi:xa]  = frame.RawImageB.astype(float)

                if perform_transformation:
                    transf = ProjectiveTransform(matrix = shift_matrix @ (self.tr_matr_cum_residual[frame_ind] @ inv_shift_matrix))
                    frame_imgA_reg = warp(frame_imgA, transf, order = int_order, preserve_range=True)
                    if self.DetB != 'None':
                        frame_imgB_reg = warp(frame_imgB, transf, order = int_order, preserve_range=True)
                else:
                    frame_imgA_reg = frame_imgA.copy()
                    if self.DetB != 'None':
                        frame_imgB_reg = frame_imgB.copy()

                if flipY:
                    frame_imgA_reg = np.flip(frame_imgA_reg, axis=0)
                    if self.DetB != 'None':
                        frame_imgB_reg = np.flip(frame_imgB_reg, axis=0)
                    xi_eval, xa_eval = eval_bounds[j, 0:2]
                    ya_eval, yi_eval = ysz - np.array(eval_bounds)[j, 2:4]
                else:
                    xi_eval, xa_eval, yi_eval, ya_eval = eval_bounds[j, :]

                frame_imgA_eval = frame_imgA_reg[yi_eval:ya_eval, xi_eval:xa_eval]
                frame_imgB_eval = frame_imgB_reg[yi_eval:ya_eval, xi_eval:xa_eval]
                frame_imgF_eval = frame_imgA_eval * (1.0 - ImgB_fraction) + frame_imgB_eval * ImgB_fraction
                print('Analyzing the Fused Image, Det B fraction = {:.4f}'.format(ImgB_fraction))
                ImageF_xSNR, ImageF_ySNR, ImageF_rSNR = Single_Image_SNR(frame_imgF_eval,
                                                                        zero_mean = zero_mean,
                                                                        extrapolate_signal=extrapolate_signal,
                                                                        edge_fraction = edge_fraction,
                                                                        aperture = aperture,
                                                                        nlags = nlags,
                                                                        save_res_png=save_res_png,
                                                                        res_fname = SNR_png_fname.replace('.png', '_ImgB_fr{:.3f}_SNR.png'.format(ImgB_fraction)),
                                                                        img_label='Fused, ImB_fr={:.4f}, frame={:d}'.format(ImgB_fraction, frame_ind))
                xSNRFs.append(ImageF_xSNR)
                ySNRFs.append(ImageF_ySNR)
                rSNRFs.append(ImageF_rSNR)

            ax.plot(frame_inds, xSNRFs, 'rd', linestyle='dashed', label='Fused Image x-SNR')
            ax.plot(frame_inds, ySNRFs, 'bd', linestyle='dashed', label='Fused Image y-SNR')
            ax.plot(frame_inds, rSNRFs, 'gd', linestyle='dashed', label='Fused Image r-SNR')

        else:
            ImgB_fraction_xSNR = 0.0
            ImgB_fraction_ySNR = 0.0
            ImgB_fraction_rSNR = 0.0
        ax.grid(True)
        ax.legend()
        ax.set_title(Sample_ID + '  ' + data_dir, fontsize=8)
        ax.set_xlabel('Frame')
        ax.set_ylabel('SNR')
        if save_res_png :
            fig_filename = os.path.join(data_dir, os.path.splitext(fnm_reg)[0]+'SNR_evaluation_mult_frame.png')
            fig.savefig(fig_filename, dpi=300)

        return ImgB_fraction_xSNR, ImgB_fraction_ySNR, ImgB_fraction_rSNR


    def evaluate_ImgB_fractions(self, ImgB_fractions, frame_inds, **kwargs):
        '''
        Calculate NCC and SNR vs Image B fraction over a set of frames.

        Parameters:
        ---------
        ImgB_fractions : list
            List of fractions to estimate the NCC and SNR.
        frame_inds : int array
            List or array array of frame indices to perform NCC / SNR evaluation.
        
        kwargs:
        ---------
        DASK_client : DASK client
            If set to empty string '' (default), local computations are performed.
        DASK_client_retries : int 
            Number of allowed automatic retries if a task fails. Default is object attribute.
        zero_mean: boolean
            If True (default), auto-correlation is zero-mean. Default is True.
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration
            if evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        extrapolate_signal : str
            Extrapolation method to find signal auto-correlation at 0-point (“Noise-Free Autocorrelation”). Options are:
                'nearest'  - nearest point (1 pixel away from center, same as NN in [1]).
                'linear'   - linear interpolation of 2-points next to center (same as FO in [1]).
                'parabolic' - parabolic interpolation of 2 point left and 2 points right (for 4-point interpolation this is the same as NN+FO in [1]).
                'gaussian'  - gaussian interpolation with number of points = aperture
                'LDR' - use Levinson-Durbin recusrsion (ACLDR in [1]).
            Default is 'parabolic'.
        edge_fraction : float
            Fraction of the full auto-correlation  range used to calculate the "zero value" (default is 0.10).
        nlags : int
            In case of 'LDR' (Levinson-Durbin recursion) nlags is the recursion order (a number of lags). Default is min(xsize/4, ysize/4).
        aperture : int
            Total number of points for gaussian interpolation. Default is 6.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        data_dir : str
            Data directory (path). Default is object attribute.
        Sample_ID : str
            Sample ID. Default is object attribute.
        invert_data : boolean
            If True - the data is inverted. Default is object attribute.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is object attribute.
        pad_edges : boolean
            If True, the frame will be padded to account for frame position and/or size changes. Default is object attribute.
        save_res_png  : boolean
            Save PNG images of the intermediate processing statistics and final registration quality check.

        Returns:
            SNRimpr_max_position, SNRimpr_max, ImgB_fractions, SNRs
        '''
        
        DASK_client = kwargs.get('DASK_client', '')
        use_DASK, status_update_address = check_DASK(DASK_client)
        if hasattr(self, "DASK_client_retries"):
            DASK_client_retries = kwargs.get("DASK_client_retries", self.DASK_client_retries)
        else:
            DASK_client_retries = kwargs.get("DASK_client_retries", 3)

        zero_mean = kwargs.get('zero_mean', True)
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        extrapolate_signal = kwargs.get('extrapolate_signal', 'parabolic')
        edge_fraction = kwargs.get("edge_fraction", 0.10)
        aperture = kwargs.get("aperture", 6)
        shape = [self.YResolution, self.XResolution]
        nlags = kwargs.get("nlags", np.min(shape)//4)
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        fnm_reg = kwargs.get("fnm_reg", self.fnm_reg)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        if hasattr(self, 'invert_data'):
            invert_data = kwargs.get("invert_data", self.invert_data)
        else:
            invert_data = kwargs.get("invert_data", False)
        if hasattr(self, 'flipY'):
            flipY = kwargs.get("flipY", self.flipY)
        else:
            flipY = kwargs.get("flipY", flipY)
        flatten_image = kwargs.get("flatten_image", False)
        save_res_png  = kwargs.get("save_res_png", self.save_res_png )
        
        save_sample_frames_png = kwargs.get("save_sample_frames_png", False )
        

        nbr = len(ImgB_fractions)
        kwargs_local ={'zbin_factor' : 1}

        test_frame = FIBSEM_frame(self.fls[frame_inds[0]], calculate_scaled_images=False)
        xi = 0
        yi = 0
        xsz = test_frame.XResolution 
        xa = xi + xsz
        ysz = test_frame.YResolution
        ya = yi + ysz

        xi_eval = xi + evaluation_box[2]
        if evaluation_box[3] > 0:
            xa_eval = xi_eval + evaluation_box[3]
        else:
            xa_eval = xa
        yi_eval = yi + evaluation_box[0]
        if evaluation_box[1] > 0:
            ya_eval = yi_eval + evaluation_box[1]
        else:
            ya_eval = ya

        br_results = []
        xSNRFs=[]
        ySNRFs=[]
        rSNRFs=[]

        for ImgB_fraction in tqdm(ImgB_fractions, desc='Evaluating Img B fractions'):
            kwargs_local['ImgB_fraction'] = ImgB_fraction
            kwargs_local['disp_res'] = False
            kwargs_local['evaluation_box'] = evaluation_box
            kwargs_local['flipY'] = flipY
            kwargs_local['invert_data'] = invert_data
            kwargs_local['disp_res'] = False
            kwargs_local['flatten_image'] = flatten_image
            br_res, br_res_xlsx = self.transform_and_save(DASK_client = DASK_client,
                                                            ftype = ftype,
                                                            save_transformed_dataset=False,
                                                            save_registration_summary=False,
                                                            frame_inds=frame_inds,
                                                            save_sample_frames_png=False,
                                                            eval_metrics = ['NCC'],
                                                            **kwargs_local)
            br_results.append(br_res)

            if invert_data:
                if test_frame.EightBit==0:
                    frame_imgA = np.negative(test_frame.RawImageA)
                    if self.DetB != 'None':
                        frame_imgB = np.negative(test_frame.RawImageB)
                else:
                    frame_imgA  =  np.uint8(255) - test_frame.RawImageA
                    if self.DetB != 'None':
                        frame_imgB  =  np.uint8(255) - test_frame.RawImageB
            else:
                frame_imgA  = test_frame.RawImageA
                if self.DetB != 'None':
                    frame_imgB  = test_frame.RawImageB
            if flipY:
                frame_imgA = np.flip(frame_imgA, axis=0)
                frame_imgB = np.flip(frame_imgB, axis=0)
                yeval_delta = ya_eval - yi_eval
                yi_eval = ysz - ya_eval
                ya_eval = yi_eval + yeval_delta

            frame_imgA_eval = frame_imgA[yi_eval:ya_eval, xi_eval:xa_eval]
            frame_imgB_eval = frame_imgB[yi_eval:ya_eval, xi_eval:xa_eval]

            frame_imgF_eval = frame_imgA_eval * (1.0 - ImgB_fraction) + frame_imgB_eval * ImgB_fraction
            ImageF_xSNR, ImageF_ySNR, ImageF_rSNR = Single_Image_SNR(frame_imgF_eval,
                                                                    zero_mean=zero_mean,
                                                                    extrapolate_signal=extrapolate_signal,
                                                                    edge_fraction = edge_fraction,
                                                                    aperture = aperture,
                                                                    nlags = nlags,
                                                                    disp_res=False,
                                                                    save_res_png=False,
                                                                    res_fname = '',
                                                                    img_label='')
            xSNRFs.append(ImageF_xSNR)
            ySNRFs.append(ImageF_ySNR)
            rSNRFs.append(ImageF_rSNR)

        fig, axs = plt.subplots(4,1, figsize=(6,11))
        fig.subplots_adjust(left=0.12, bottom=0.06, right=0.99, top=0.96, wspace=0.25, hspace=0.24)
        try:
            ncc0 = (br_results[0])['NCC']
        except:
            ncc0 = (br_results[0])['Image NCC']
        SNR0 = ncc0 / (1-ncc0)
        SNRimpr_cc = []
        SNRs = []

        for j, (ImgB_fraction, br_result) in enumerate(zip(ImgB_fractions, br_results)):
            my_col = plt.get_cmap("gist_rainbow_r")((nbr-j)/(nbr-1))
            try:
                ncc = br_result['NCC']
            except:
                ncc = br_result['Image NCC']
            SNR = ncc / (1.0-ncc)
            frames_local = br_result['Frame']
            axs[0].plot(frames_local, SNR, color=my_col, label = 'ImgB fraction = {:.2f}'.format(ImgB_fraction))
            axs[1].plot(frames_local, SNR/SNR0, color=my_col, label = 'ImgB fraction = {:.2f}'.format(ImgB_fraction))
            SNRimpr_cc.append(np.mean(SNR/SNR0))
            SNRs.append(np.mean(SNR))

        SNRimpr_ac = np.array(rSNRFs) / rSNRFs[0]

        SNRimpr_cc_max = np.max(SNRimpr_cc)
        SNRimpr_cc_max_ind = np.argmax(SNRimpr_cc)
        ImgB_fraction_max = ImgB_fractions[SNRimpr_cc_max_ind]
        xi = max(0, (SNRimpr_cc_max_ind-3))
        xa = min((SNRimpr_cc_max_ind+3), len(ImgB_fractions))
        ImgB_fr_range = ImgB_fractions[xi : xa]
        SNRimpr_cc_range = SNRimpr_cc[xi : xa]
        popt = np.polyfit(ImgB_fr_range, SNRimpr_cc_range, 2)
        SNRimpr_cc_fit_max_pos = -0.5 * popt[1] / popt[0]
        ImgB_fr_fit_cc = np.linspace(ImgB_fr_range[0], ImgB_fr_range[-1], 21)
        SNRimpr_cc_fit = np.polyval(popt, ImgB_fr_fit_cc)
        if popt[0] < 0 and SNRimpr_cc_fit_max_pos > ImgB_fractions[0] and SNRimpr_cc_fit_max_pos<ImgB_fractions[-1]:
            SNRimpr_cc_max_position = SNRimpr_cc_fit_max_pos
            SNRimpr_cc_max = np.polyval(popt, SNRimpr_cc_max_position)
        else: 
            SNRimpr_cc_max_position = ImgB_fraction_max

        SNRimpr_ac_max = np.max(SNRimpr_ac)
        SNRimpr_ac_max_ind = np.argmax(SNRimpr_ac)
        ImgB_fraction_max = ImgB_fractions[SNRimpr_ac_max_ind]
        xi = max(0, (SNRimpr_ac_max_ind-3))
        xa = min((SNRimpr_ac_max_ind+3), len(ImgB_fractions))
        ImgB_fr_range = ImgB_fractions[xi : xa]
        SNRimpr_ac_range = SNRimpr_ac[xi : xa]
        popt = np.polyfit(ImgB_fr_range, SNRimpr_ac_range, 2)
        SNRimpr_ac_fit_max_pos = -0.5 * popt[1] / popt[0]
        ImgB_fr_fit_ac = np.linspace(ImgB_fr_range[0], ImgB_fr_range[-1], 21)
        SNRimpr_ac_fit = np.polyval(popt, ImgB_fr_fit_ac)
        if popt[0] < 0 and SNRimpr_ac_fit_max_pos > ImgB_fractions[0] and SNRimpr_ac_fit_max_pos<ImgB_fractions[-1]:
            SNRimpr_ac_max_position = SNRimpr_ac_fit_max_pos
            SNRimpr_ac_max = np.polyval(popt, SNRimpr_ac_max_position)
        else: 
            SNRimpr_ac_max_position = ImgB_fraction_max
            
        fs=10
        axs[0].grid(True)
        axs[0].set_ylabel('Frame-to-Frame SNR', fontsize=fs)
        axs[0].set_xlabel('Frame', fontsize=fs)  
        axs[0].legend(fontsize=fs-1)
        axs[0].set_title(Sample_ID + '  ' + data_dir, fontsize=fs)
        axs[1].grid(True)
        axs[1].set_ylabel('Frame-to-Frame SNR Improvement', fontsize=fs)
        axs[1].set_xlabel('Frame', fontsize=fs)
        axs[2].plot(ImgB_fractions, rSNRFs, 'rd', label='Data (auto-correlation)')
        axs[2].grid(True)
        axs[2].set_ylabel('Auto-Corr SNR', fontsize=fs)
        axs[3].plot(ImgB_fractions, SNRimpr_cc, 'cs', label='Data (cross-corr.)')
        axs[3].plot(ImgB_fr_fit_cc, SNRimpr_cc_fit, 'b', label='Fit (cross-corr.)')
        axs[3].plot(SNRimpr_cc_max_position, SNRimpr_cc_max, 'bx', markersize=10, label='Max SNR Impr. (cc)')
        axs[3].text(0.4, 0.35, 'Max CC SNR Improvement={:.3f}'.format(SNRimpr_cc_max), transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.4, 0.25, '@ Img B Fraction ={:.3f}'.format(SNRimpr_cc_max_position), transform=axs[3].transAxes, fontsize=fs)
        axs[3].plot(ImgB_fractions, SNRimpr_ac, 'rd', label='Data (auto-corr.)')
        axs[3].plot(ImgB_fr_fit_ac, SNRimpr_ac_fit, 'magenta', label='Fit (auto-corr.)')
        axs[3].plot(SNRimpr_ac_max_position, SNRimpr_ac_max, 'mx', markersize=10, label='Max SNR Impr. (ac)')
        axs[3].text(0.4, 0.15, 'Max AC SNR Improvement={:.3f}'.format(SNRimpr_ac_max), transform=axs[3].transAxes, fontsize=fs)
        axs[3].text(0.4, 0.05, '@ Img B Fraction ={:.3f}'.format(SNRimpr_ac_max_position), transform=axs[3].transAxes, fontsize=fs)
        axs[3].legend(fontsize=fs-2, loc='upper left')
        axs[3].grid(True)
        axs[3].set_ylabel('Mean SNR improvement', fontsize=fs)
        axs[3].set_xlabel('Image B fraction', fontsize=fs)

        if save_res_png :
            fname_image = os.path.join(data_dir, os.path.splitext(fnm_reg)[0]+'_SNR_vs_ImgB_ratio_evaluation.png')
            fig.savefig(fname_image, dpi=300)

        return SNRimpr_cc_max_position, SNRimpr_cc_max, ImgB_fractions, SNRs, rSNRFs


    def estimate_resolution_blobs_2D(self, **kwargs):
        '''
        Estimate transitions in the image, uses select_blobs_LoG_analyze_transitions(image, **kwargs). gleb.shtengel@gmail.com  06/2023 

        kwargs:
        ---------
        DASK_client : DASK client. If set to empty string '' (default), local computations are performed
        DASK_client_retries : int
            Number of allowed automatic retries if a task fails. Default is 3.
        image_name : str
            Options are: 'RawImageA' (default), 'RawImageB', 'ImageA', 'ImageB'.
        frame_inds : list of int
            Array or list of frame indices to use to display the evaluation box. Default is [nfrs//10, nfrs//2, nfrs//10*9].
        evaluation_box : list of 4 int
            evaluation_box = [top, height, left, width] boundaries of the box used for evaluating the image registration.
            If evaluation_box is not set or evaluation_box = [0, 0, 0, 0], the entire image is used.
        sliding_evaluation_box : boolean
            if True, then the evaluation box will be linearly interpolated between sliding_evaluation_box and stop_evaluation_box.
        start_evaluation_box : list of 4 int
            see above.
        stop_evaluation_box : list of 4 int
            see above.
        ftype : int
            File type (0 - Shan Xu's .dat, 1 - tif). Default is object attribute.
        data_dir : str
            Data directory (path). Default is object attribute.
        Sample_ID : str
            Sample ID. Default is object attribute.
        perform_transformation : boolean
            If True - the data is transformed using existing cumulative transformation matrix. If False - the data is not transformed. Default is False.
        int_order : int
            The order of interpolation (when transforming the data). Default is object attribute.
                The order has to be in the range 0-5:
                    0: Nearest-neighbor
                    1: Bi-linear (default)
                    2: Bi-quadratic
                    3: Bi-cubic
                    4: Bi-quartic
                    5: Bi-quintic
        invert_data : boolean
            If True - the data is inverted. Default is object attribute.
        flipY : boolean
            If True, the data will be flipped along Y-axis. Default is object attribute.
        pad_edges : boolean
            If True, the frame will be padded to account for frame position and/or size changes. Default is object attribute.
        zbin_factor : int
            Binning factor along Z-axis. Default is object attribute.
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
            Pixel size in nm. Default is the value of object attribute PixelSize. 
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
        title : str
            Title string.
        nbins : int
            Number of bins for histogram. Default is 64.
        save_data_xlsx : boolean
            Save the data into Excel workbook. Default is True.
        results_file_xlsx : file name for Excel workbook to save the results. Default is derived from self.fnm_reg +'_2D_blob_analysis_results.xlsx'
            
        Returns: XY_transitions ; array[len(frame_inds), 4]
            array consists of lines - one line for each frame in frame_inds
            each line has 4 elements: [Xpt1, Xpt2, Ypt1, Ypt2]
        '''
        DASK_client = kwargs.get('DASK_client', '')
        DASK_client_retries = kwargs.get('DASK_client_retries', 3)
        use_DASK, status_update_address = check_DASK(DASK_client)

        image_name = kwargs.get("image_name", 'ImageA')
        evaluation_box = kwargs.get("evaluation_box", [0, 0, 0, 0])
        sliding_evaluation_box = kwargs.get("sliding_evaluation_box", False)
        start_evaluation_box = kwargs.get("start_evaluation_box", [0, 0, 0, 0])
        stop_evaluation_box = kwargs.get("stop_evaluation_box", [0, 0, 0, 0])
        ftype = kwargs.get("ftype", self.ftype)
        data_dir = kwargs.get("data_dir", self.data_dir)
        Sample_ID = kwargs.get("Sample_ID", self.Sample_ID)
        
        if hasattr(self, 'perform_transformation'):
            perform_transformation = kwargs.get("perform_transformation", self.perform_transformation)
        else:
            perform_transformation = kwargs.get("perform_transformation", False)
        kwargs['perform_transformation'] = perform_transformation
        if hasattr(self, 'int_order'):
            int_order = kwargs.get("int_order", self.int_order)
        else:
            int_order = kwargs.get("int_order", 1)
        kwargs['int_order'] = int_order
        if hasattr(self, 'invert_data'):
            invert_data = kwargs.get("invert_data", self.invert_data)
        else:
            invert_data = kwargs.get("invert_data", False)
        kwargs['invert_data'] = invert_data
        if hasattr(self, 'flipY'):
            flipY = kwargs.get("flipY", self.flipY)
        else:
            flipY = kwargs.get("flipY", False)
        kwargs['flipY'] = flipY
        if hasattr(self, 'pad_edges'):
            pad_edges = kwargs.get("pad_edges", self.pad_edges)
        else:
            pad_edges = kwargs.get("pad_edges", False)
        kwargs['pad_edges'] = pad_edges
        if hasattr(self, 'zbin_factor'):
            zbin_factor = kwargs.get("zbin_factor", self.zbin_factor)
        else:
            zbin_factor = kwargs.get("zbin_factor", 1)
        kwargs['zbin_factor'] = zbin_factor
        
        min_sigma = kwargs.get('min_sigma', 1.0)
        max_sigma = kwargs.get('max_sigma', 1.0)
        
        overlap = kwargs.get('overlap', 0.1)
        subset_size = kwargs.get('subset_size', 16)     # blob analysis window size in pixels
        dx2=subset_size//2
        pixel_size = kwargs.get('pixel_size', self.PixelSize)
        bounds = kwargs.get('bounds', [0.37, 0.63])
        bands = kwargs.get('bands', [5, 5, 5])        # bands for finding left minimum, mid (peak), and right minimum
        min_thr = kwargs.get('min_thr', 0.4)        #threshold for identifying 'good' transition (bottom < min_thr* top)
        transition_low_limit = kwargs.get('transition_low_limit', 0.0)
        transition_high_limit = kwargs.get('transition_high_limit', 10.0)
        save_data_xlsx = kwargs.get('save_data_xlsx', True)

        if hasattr(self, 'fnm_reg'):
            default_results_file_xlsx = os.path.join(data_dir, os.path.splitext(self.fnm_reg)[0]+'_2D_blob_analysis_results.xlsx')
        else:
            default_results_file_xlsx = os.path.join(data_dir, 'Dataset_2D_blob_analysis_results.xlsx')
        results_file_xlsx = kwargs.get('results_file_xlsx', default_results_file_xlsx)
        
        if use_DASK:
            verbose = False
            disp_res = False
        else:
            verbose = kwargs.get('verbose', False)
            disp_res = kwargs.get('disp_res', False)

        title = kwargs.get('title', '')
        nbins = kwargs.get('nbins', 64)

        fls = np.array(self.fls)
        nfrs = len(fls)
        default_indices = np.arange(nfrs)
        frame_inds = kwargs.get("frame_inds", default_indices)
        if verbose:
            print('Will analyze frames with inds:', frame_inds)
            print('Frame files:', np.array(fls)[frame_inds])
        fls_df = pd.DataFrame(fls[frame_inds], columns = ['Frame Filename'], index = None)
        fls_df.insert(0, 'Frame', frame_inds)

        shape = [self.YResolution, self.XResolution]
        shapes = [self.YResolutions, self.XResolutions]

        vmin = 0.05
        if image_name == 'ImageA':
            vmin, vmax = get_min_max_thresholds(FIBSEM_frame(self.fls[frame_inds[0]], calculate_scaled_images=True).ImageA, thr_min=0.2, disp_res=False, save_res=False)
        if image_name == 'ImageB':
            vmin, vmax = get_min_max_thresholds(FIBSEM_frame(self.fls[frame_inds[0]], calculate_scaled_images=True).ImageB, thr_min=0.2, disp_res=False, save_res=False)
        if image_name == 'RawImageA':
            vmin, vmax = get_min_max_thresholds(FIBSEM_frame(self.fls[frame_inds[0]], calculate_scaled_images=False).RawImageA, thr_min=0.2, disp_res=False, save_res=False)
        if image_name == 'RawImageB':
            vmin, vmax = get_min_max_thresholds(FIBSEM_frame(self.fls[frame_inds[0]], calculate_scaled_images=False).RawImageB, thr_min=0.2, disp_res=False, save_res=False)
        threshold = kwargs.get('threshold', vmin/10.0)
        if verbose:
            print('Will use threshold : {:.4f}'.format(threshold))

        eval_bounds = set_eval_bounds(shape, evaluation_box,
            start_evaluation_box = start_evaluation_box,
            stop_evaluation_box = stop_evaluation_box,
            sliding_evaluation_box = sliding_evaluation_box,
            pad_edges = pad_edges,
            perform_transformation = perform_transformation,
            tr_matr =  self.tr_matr_cum_residual,
            frame_inds = frame_inds)

        if pad_edges and perform_transformation:
            xi, yi, xsz, ysz = determine_sizes_and_offsets(shapes, self.tr_matr_cum_residual)
            #xi, yi, padx, pady = determine_pad_offsets(shape, self.tr_matr_cum_residual)
            shift_matrix = np.array([[1.0, 0.0, xi],
                                 [0.0, 1.0, yi],
                                 [0.0, 0.0, 1.0]]).astype(float)
            inv_shift_matrix = np.linalg.inv(shift_matrix)
        else:
            xi = 0
            yi = 0
            xsz = shape[1]
            ysz = shape[0]
            shift_matrix = np.eye(3,3)
            inv_shift_matrix = np.eye(3,3)
        offsets_sizes = [xi, yi, xsz, ysz]
        kwargs['offsets_sizes'] = offsets_sizes

        papams_blob_analysis = []
        results_2D = []

        transformation_matrix = np.array(self.tr_matr_cum_residual)[frame_inds]
        columns=['T00 (Sxx)', 'T01 (Sxy)', 'T02 (Tx)',  
                 'T10 (Syx)', 'T11 (Syy)', 'T12 (Ty)', 
                 'T20 (0.0)', 'T21 (0.0)', 'T22 (1.0)']
        tr_mx_df = pd.DataFrame(transformation_matrix.reshape((len(transformation_matrix), 9)), columns = columns, index = None)
        eval_bounds_df = pd.DataFrame(np.array(eval_bounds), columns = ['xi_eval', 'xa_eval', 'yi_eval', 'ya_eval'], index = None)
        fls_info = pd.concat([fls_df, eval_bounds_df, tr_mx_df], axis=1)

        for j, frame_ind in enumerate(tqdm(frame_inds, desc='Building the Parameter Sets Analyzing Resolution using Blobs ', display=verbose)):
            params_single = [fls, frame_ind, ftype, image_name, eval_bounds[j], offsets_sizes, invert_data, flipY, zbin_factor, perform_transformation, self.tr_matr_cum_residual[frame_ind], int_order, pad_edges, min_sigma, max_sigma, threshold,  overlap, pixel_size, subset_size, bounds, bands, min_thr, transition_low_limit, transition_high_limit, nbins, verbose, disp_res, False]
            papams_blob_analysis.append(params_single)

        if use_DASK:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using DASK distributed')
            futures = DASK_client.map(select_blobs_LoG_analyze_transitions_2D_dataset, papams_blob_analysis, retries = DASK_client_retries)
            results_2D = DASK_client.gather(futures)

        else:
            print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Using Local Computation')

            for j, params_single in enumerate(tqdm(papams_blob_analysis, desc='Analyzing Resolution using Blobs', display = verbose)):
                if verbose:
                    print('Analyzing file: ', fls[params_single[1]])
                results_2D.append(select_blobs_LoG_analyze_transitions_2D_dataset(params_single))

        frame_inds = np.concatenate(np.array(results_2D, dtype=object)[:, 0], axis=0)
        error_flags = np.concatenate(np.array(results_2D, dtype=object)[:, 1], axis=0)
        blobs_LoG = np.concatenate(np.array(results_2D, dtype=object)[:, 2], axis=0)
        tr_results = np.concatenate(np.array(results_2D, dtype=object)[:, 3], axis=0)

        if save_data_xlsx:
            xlsx_writer = pd.ExcelWriter(results_file_xlsx, engine='xlsxwriter')
            trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
            columns=['Frame', 'Y', 'X', 'R', 'Amp',
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

            if verbose:
                print(time.strftime('%Y/%m/%d  %H:%M:%S')+'   Saving the results into file:  ' + results_file_xlsx)

            transition_results = pd.DataFrame(np.column_stack((frame_inds, blobs_LoG, tr_results, error_flags)), columns = columns, index = None)
            transition_results.to_excel(xlsx_writer, index=None, sheet_name='Transition analysis results')
            kwargs_info = pd.DataFrame([kwargs]).T # prepare to be save in transposed format
            kwargs_info.to_excel(xlsx_writer, header=False, sheet_name='kwargs Info')
            fls_info.to_excel(xlsx_writer, index=None, sheet_name='Frame Filenames')
            fexts =['_{:.0f}{:.0f}pts'.format(bounds[0]*100, bounds[1]*100), '_{:.0f}{:.0f}slp'.format(bounds[0]*100, bounds[1]*100)]
            sheet_names = ['{:.0f}%-{:.0f}% summary (pts)'.format(bounds[0]*100, bounds[1]*100),
                '{:.0f}%-{:.0f}% summary (slopes)'.format(bounds[0]*100, bounds[1]*100)]
            #xlsx_writer.save()
            xlsx_writer.close()
        
        # return results_2D
        return results_file_xlsx, frame_inds, error_flags, blobs_LoG, tr_results
        

def plot_2D_blob_results(results_xlsx, **kwargs):
    '''
    Generates the summary plot based on xlsx file created by estimate_resolution_blobs_2D
    estimate_resolution_blobs_2D analyzes the transitions in the image, uses select_blobs_LoG_analyze_transitions(frame_eval, **kwargs). gleb.shtengel@gmail.com  04/2025 

    Parameters:
    ---------
    results_file_xlsx : file name of Excel workbook of the results created by estimate_resolution_blobs_2D
    
    kwargs:
    ---------
    save_png : boolean
        Save the image into PNG file. Default is False.
    save_fname : string
        File name for to save the image. Default is created by replace('.xlsx', '_2D_blob_analysis_results.png').
    nbins : int
        Number of bins for histogram. Default is 64.
    verbose : boolean
        Print the outputs. Default is False.
    title : str
        Title. Default is 'Blobs determined by Laplasian of Gaussians'.
    '''
    save_png = kwargs.get('save_png', False)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_2D_blob_analysis_results.png'))
    nbins = kwargs.get('nbins', 64)
    title = kwargs.get('title', 'Blobs determined by Laplasian of Gaussians')

    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    pixel_size = saved_kwargs.get("pixel_size", 0.0)
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    perform_transformation =  saved_kwargs.get("perform_transformation", False)
    transition_low_limit = saved_kwargs.get("transition_low_limit", 0.0)
    transition_high_limit = saved_kwargs.get("transition_high_limit", 10.0)
        
    xs=7.0
    ys = xs*1.5
    text_col = 'brown'
    text_fs = 12
    axis_label_fs = 10
    table_fs = 12
    caption_fs = 7
    
    trans_str = '{:.2f} to {:.2f} transition (nm)'.format(bounds[0], bounds[1])
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
    XYpt_selected = [Xpt1, Xpt2, Ypt1, Ypt2]
    Xpt_selected = [Xpt1, Xpt2]
    Ypt_selected = [Ypt1, Ypt2]
    Xslp_selected = [Xslp1, Xslp2]
    Yslp_selected = [Yslp1, Yslp2]
    tr_median = np.median(XYpt_selected)
    tr_mean = np.mean(XYpt_selected)
    tr_std = np.std(XYpt_selected)
    
    fig, axs = plt.subplots(2, 1, figsize=(xs,ys))
    fig.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.97, wspace=0.02, hspace=0.12)
    ax1, ax2 = axs
    ax1.set_title(title, fontsize=text_fs)
    ax1.text(0.6, 0.55, '# of blobs: {:d}'.format(len(Xpt1)), transform=ax1.transAxes, color=text_col, fontsize=text_fs)
    ax1.text(0.6, 0.50, '{:.0f}% - {:.0f}% Transitions'.format(bounds[0]*100, bounds[1]*100), transform=ax1.transAxes, color=text_col, fontsize=text_fs)
    ax1.text(0.6, 0.45, 'Pixel Size (nm): {:.3f}'.format(pixel_size), transform=ax1.transAxes, color=text_col, fontsize=text_fs)
    ax1.text(0.6, 0.40, 'Median value (nm): {:.3f}'.format(tr_median), transform=ax1.transAxes, color=text_col, fontsize=text_fs)
    ax1.text(0.6, 0.35, 'Mean value (nm): {:.3f}'.format(tr_mean), transform=ax1.transAxes, color=text_col, fontsize=text_fs)
    ax1.text(0.6, 0.30, 'STD (nm):       {:.3f}'.format(tr_std), transform=ax1.transAxes, color=text_col, fontsize=text_fs)
    
    tr_sets = [[Xpt_selected, Ypt_selected],
          [Xslp_selected, Yslp_selected]]
    fexts =['_{:.0f}{:.0f}pts'.format(bounds[0]*100, bounds[1]*100), '_{:.0f}{:.0f}slp'.format(bounds[0]*100, bounds[1]*100)]
    xaxis_labels = ['{:.0f}%-{:.0f}% Transition (pts) (nm)'.format(bounds[0]*100, bounds[1]*100),
        '{:.0f}%-{:.0f}% Transition (slope) (nm)'.format(bounds[0]*100, bounds[1]*100)]
    hranges = [(transition_low_limit, transition_high_limit), 
           (transition_low_limit, transition_high_limit)]  # histogram range for the transition distance (in nm))
       
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
        cell_text = [['{:.3f}'.format(d) for d in dd] for dd in hst_data]

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
        try:
            table_cells = table_props['child_artists']
        except:
            table_cells = table_props['children']

        for j, cell in enumerate(table_cells[0:n_cols*n_rows]):
            cell.get_text().set_color(cols[j//n_cols])
            cell.get_text().set_fontsize(table_fs)
        for j, cell in enumerate(table_cells[n_cols*(n_rows+1):]):
            cell.get_text().set_color(cols[j])
        for cell in table_cells[n_cols*n_rows:]:
        #    cell.get_text().set_fontweight('bold')
            cell.get_text().set_fontsize(table_fs)
    
    if save_png:
        ax2.text(-0.1, -0.15, save_fname, transform=ax2.transAxes, fontsize=caption_fs)
        fig.savefig(save_fname, dpi=300)


def plot_2D_blob_examples(results_xlsx, **kwargs):
    '''
    Generates a figure with blob examples based on xlsx file created by estimate_resolution_blobs_2D
    estimate_resolution_blobs_2D analyzes the transitions in the image, uses select_blobs_LoG_analyze_transitions(frame_eval, **kwargs). gleb.shtengel@gmail.com  04/2025 

    Parameters:
    ---------
    results_file_xlsx : file name of Excel workbook of the results created by estimate_resolution_blobs_2D
    
    kwargs:
    ---------
    save_png : boolean
        save the image into PNG file. Default is False.
    save_fname : string
        file name for to save the image. Default is created by replace('.xlsx', '_2D_blob_examples.png').
    verbose : boolean
        Print the outputs. Default is False.
    '''
    save_png = kwargs.get('save_png', False)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_2D_blob_examples.png'))
    verbose = kwargs.get('verbose', False)

    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    pixel_size = saved_kwargs.get("pixel_size", 1.0)
    subset_size = saved_kwargs.get("subset_size", 2.0)
    dx2 = subset_size//2
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    bands = saved_kwargs.get("bands", [3, 2, 3])
    image_name = saved_kwargs.get("image_name", 'ImageA')
    calculate_scaled_images = (image_name == 'ImageA') or (image_name == 'ImageB')
    perform_transformation =  saved_kwargs.get("perform_transformation", False)
    pad_edges =  saved_kwargs.get("pad_edges", True)
    ftype =  saved_kwargs.get("ftype", 0)
    zbin_factor =  saved_kwargs.get("zbin_factor", 1)
    flipY = saved_kwargs.get("flipY", False)
    invert_data = saved_kwargs.get("invert_data", False)
    int_order =  saved_kwargs.get("int_order", 1)
    offsets =  saved_kwargs.get("offsets", [0, 0, 0, 0])
    
    fls_info = pd.read_excel(results_xlsx, sheet_name='Frame Filenames')
    fls = fls_info['Frame Filename']
    frame_inds = fls_info['Frame']
    transformation_matrix = np.vstack((fls_info['T00 (Sxx)'],
                         fls_info['T01 (Sxy)'],
                         fls_info['T02 (Tx)'],
                         fls_info['T10 (Syx)'],
                         fls_info['T11 (Syy)'],
                         fls_info['T12 (Ty)'],
                         fls_info['T20 (0.0)'],
                         fls_info['T21 (0.0)'],
                         fls_info['T22 (1.0)'])).T.reshape((len(fls_info['T00 (Sxx)']), 3, 3))

    xs=16.0
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
    X_selected = np.array(X)[error_flags==0]
    Y_selected = np.array(Y)[error_flags==0]
    frames = int_results['Frame']
    frames_selected = np.array(frames)[error_flags==0]
    Xs = np.concatenate((X_selected[0:3], X_selected[-3:]))
    Ys = np.concatenate((Y_selected[0:3], Y_selected[-3:]))
    Fs = np.concatenate((frames_selected[0:3], frames_selected[-3:]))

    xt = 0.0
    yt = 1.5 
    clr_x = 'green'
    clr_y = 'blue'
    fig, axs = plt.subplots(4,3, figsize=(xs, ys))
    fig.subplots_adjust(left=0.02, bottom=0.04, right=0.99, top=0.99, wspace=0.15, hspace=0.12)

    ax_maps = [axs[0,0], axs[0,1], axs[0,2], axs[2,0], axs[2,1], axs[2,2]]
    ax_profiles = [axs[1,0], axs[1,1], axs[1,2], axs[3,0], axs[3,1], axs[3,2]]

    for j, x in enumerate(tqdm(Xs, desc='Generating images/plots for the sample 2D blobs')):
        local_ind = int(np.argwhere(np.array(fls_info['Frame']) == Fs[j]))
        fl_info = fls_info[fls_info['Frame'] == Fs[j]]
        fl = fl_info['Frame Filename'].values[0]
        #print(Fs[j], fl, local_ind)
        yi_eval = fl_info['yi_eval'].values[0]
        ya_eval = fl_info['ya_eval'].values[0]
        xi_eval = fl_info['xi_eval'].values[0]
        xa_eval = fl_info['xa_eval'].values[0]

        frame = FIBSEM_frame(fl, ftype=ftype, calculate_scaled_images = calculate_scaled_images)
        shape = [frame.YResolution, frame.XResolution]
        if pad_edges and perform_transformation:
            xi, yi, padx, pady = offsets
            shift_matrix = np.array([[1.0, 0.0, xi],
                                     [0.0, 1.0, yi],
                                     [0.0, 0.0, 1.0]]).astype(float)
            inv_shift_matrix = np.linalg.inv(shift_matrix)
        else:
            padx = 0
            pady = 0
            xi = 0
            yi = 0
            shift_matrix = np.eye(3,3)
            inv_shift_matrix = np.eye(3,3)
        xsz = frame.XResolution + padx
        ysz = frame.YResolution + pady
        xa = xi+frame.XResolution
        ya = yi+frame.YResolution
  
        frame_img = np.zeros((ysz, xsz), dtype=float)
        frame_eval = np.zeros(((ya_eval-yi_eval), (xa_eval-xi_eval)), dtype=float)

        for jk in np.arange(zbin_factor):
            if jk>0:
                local_ind = int(np.squeeze(np.argwhere(np.array(fls_info['Frame']) == Fs[j+jk])))
                fl_info = fls_info[fls_info['Frame'] == Fs[j+jk]]
                fl = fl_info['Frame Filename'].values[0]
                frame = FIBSEM_frame(fl, ftype=ftype, calculate_scaled_images = calculate_scaled_images)
                yi_eval = fl_info['yi_eval'].values[0]
                ya_eval = fl_info['ya_eval'].values[0]
                xi_eval = fl_info['xi_eval'].values[0]
                xa_eval = fl_info['xa_eval'].values[0]
            if invert_data:
                if frame.DetB != 'None':
                    if image_name == 'RawImageB':
                        frame_img[yi:ya, xi:xa] = np.negative(frame.RawImageB.astype(float))
                    if image_name == 'ImageB':
                        frame_img[yi:ya, xi:xa] = np.negative(frame.ImageB.astype(float))
                if image_name == 'RawImageA':
                    frame_img[yi:ya, xi:xa] = np.negative(frame.RawImageA.astype(float))
                else:
                    frame_img[yi:ya, xi:xa] = np.negative(frame.ImageA.astype(float))
            else:
                if frame.DetB != 'None':
                    if image_name == 'RawImageB':
                        frame_img[yi:ya, xi:xa]  = frame.RawImageB.astype(float)
                    if image_name == 'ImageB':
                        frame_img[yi:ya, xi:xa]  = frame.ImageB.astype(float)
                if image_name == 'RawImageA':
                    frame_img[yi:ya, xi:xa]  = frame.RawImageA.astype(float)
                else:
                    frame_img[yi:ya, xi:xa]  = frame.ImageA.astype(float)
            if perform_transformation:
                transf = ProjectiveTransform(matrix = shift_matrix @ (transformation_matrix[local_ind] @ inv_shift_matrix))
                frame_img_reg = warp(frame_img, transf, order = int_order, preserve_range=True)
            else:
                frame_img_reg = frame_img.copy()
            if flipY:
                frame_img_reg = np.flip(frame_img_reg, axis=0)

            frame_eval += frame_img_reg[yi_eval:ya_eval, xi_eval:xa_eval]
        if zbin_factor > 1:
            frame_eval /= zbin_factor
        
        y = Ys[j]
        xx = int(x)
        yy = int(y)
        subset = frame_eval[yy-dx2:yy+dx2, xx-dx2:xx+dx2]
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
        #print(np.shape(amp_x), np.shape(amp_y), np.shape(amp_z))
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



def plot_2D_blob_examples_single(img, results_xlsx, **kwargs):
    '''
    Generates a figure with blob examples based on xlsx file created by estimate_resolution_blobs_2D.
    Used in the case if a single image was used in blob analysis.
    estimate_resolution_blobs_2D analyzes the transitions in the image, uses select_blobs_LoG_analyze_transitions(frame_eval, **kwargs). gleb.shtengel@gmail.com  04/2025 

    Parameters:
    ---------
    img : 2D array
        image
    results_file_xlsx : file name of Excel workbook of the results created by estimate_resolution_blobs_2D
    
    kwargs:
    ---------
    save_png : boolean
        Save the image into PNG file. Default is False.
    save_fname : string
        File name for to save the image. Default is generated by replace('.xlsx', '_2D_blob_examples.png').
    verbose : boolean
        Print the outputs. Default is False.
    '''
    save_png = kwargs.get('save_png', False)
    save_fname = kwargs.get('save_fname', results_xlsx.replace('.xlsx', '_2D_blob_examples.png'))
    verbose = kwargs.get('verbose', False)

    saved_kwargs = read_kwargs_xlsx(results_xlsx, 'kwargs Info')
    pixel_size = saved_kwargs.get("pixel_size", 1.0)
    subset_size = saved_kwargs.get("subset_size", 2.0)
    dx2 = subset_size//2
    top_text = saved_kwargs.get("top_text", '')
    bounds = saved_kwargs.get("bounds", [0.0, 0.0])
    bands = saved_kwargs.get("bands", [3, 2, 3])
    image_name = saved_kwargs.get("image_name", 'ImageA')
    calculate_scaled_images = (image_name == 'ImageA') or (image_name == 'ImageB')
    perform_transformation =  saved_kwargs.get("perform_transformation", False)
    pad_edges =  saved_kwargs.get("pad_edges", True)
    ftype =  saved_kwargs.get("ftype", 0)
    zbin_factor =  saved_kwargs.get("zbin_factor", 1)
    flipY = saved_kwargs.get("flipY", False)
    invert_data = saved_kwargs.get("invert_data", False)
    int_order =  saved_kwargs.get("int_order", 1)
    offsets =  saved_kwargs.get("offsets", [0, 0, 0, 0])
    
    xs=16.0
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
    X_selected = np.array(X)[error_flags==0]
    Y_selected = np.array(Y)[error_flags==0]
    Xs = np.concatenate((X_selected[0:3], X_selected[-3:]))
    Ys = np.concatenate((Y_selected[0:3], Y_selected[-3:]))

    xt = 0.0
    yt = 1.5 
    clr_x = 'green'
    clr_y = 'blue'
    fig, axs = plt.subplots(4,3, figsize=(xs, ys))
    fig.subplots_adjust(left=0.02, bottom=0.04, right=0.99, top=0.99, wspace=0.15, hspace=0.12)

    ax_maps = [axs[0,0], axs[0,1], axs[0,2], axs[2,0], axs[2,1], axs[2,2]]
    ax_profiles = [axs[1,0], axs[1,1], axs[1,2], axs[3,0], axs[3,1], axs[3,2]]

    for j, x in enumerate(tqdm(Xs, desc='Generating images/plots for the sample 2D blobs')):
        ysz, xsz = img.shape
        xi = 0
        yi = 0
        xa = xi + xsz
        ya = yi + ysz

        y = Ys[j]
        xx = int(x)
        yy = int(y)
        subset = img[yy-dx2:yy+dx2, xx-dx2:xx+dx2].astype(float)
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
        #print(np.shape(amp_x), np.shape(amp_y), np.shape(amp_z))
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