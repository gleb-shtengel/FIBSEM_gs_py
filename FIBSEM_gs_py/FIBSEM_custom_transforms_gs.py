import numpy as np

import skimage
#print(skimage.__version__)
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform, warp


def _center_and_normalize_points_gs(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.
    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.
    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(D).
    If the points are all identical, the returned values will contain nan.

    Parameters:
    ----------
    points : (N, D) array
        The coordinates of the image points.
    Returns:
    -------
    matrix : (D+1, D+1) array
        The transformation matrix to obtain the new points.
    new_points : (N, D) array
        The transformed image points.
    References
    ----------
    .. [1] Hartley, Richard I. "In defense of the eight-point algorithm."
           Pattern Analysis and Machine Intelligence, IEEE Transactions on 19.6
           (1997): 580-593.
    """
    n, d = points.shape
    centroid = np.mean(points, axis=0)

    centered = points - centroid
    rms = np.sqrt(np.sum(centered ** 2) / n)

    # if all the points are the same, the transformation matrix cannot be
    # created. We return an equivalent matrix with np.nans as sentinel values.
    # This obviates the need for try/except blocks in functions calling this
    # one, and those are only needed when actual 0 is reached, rather than some
    # small value; ie, we don't need to worry about numerical stability here,
    # only actual 0.
    if rms == 0:
        return np.full((d + 1, d + 1), np.nan), np.full_like(points, np.nan)

    norm_factor = np.sqrt(d) / rms

    part_matrix = norm_factor * np.concatenate(
            (np.eye(d), -centroid[:, np.newaxis]), axis=1
            )
    matrix = np.concatenate(
            (part_matrix, [[0,] * d + [1]]), axis=0
            )

    points_h = np.row_stack([points.T, np.ones(n)])

    new_points_h = (matrix @ points_h).T

    new_points = new_points_h[:, :d]
    new_points /= new_points_h[:, d:]

    return matrix, new_points


def determine_regularized_affine_transform(src_pts, dst_pts, l2_matrix = None, targ_vector = None):
    """
    Estimate N-D affine transformation with regularization from a set of corresponding points.
    ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

        We can determine the over-, well- and under-determined parameters
        with the total least-squares method.
        Number of source and destination coordinates must match.
        The transformation is defined as:
            X = (a0*x + a1*y + a2) 
            Y = (b0*x + b1*y + b2)
        This is regularized Affine estimation - it is regularized so that the penalty is for deviation from a target (default target is rigid shift) transformation
        a0 =1, a1=0, b0=0, b1=1 are parameters for target (shift) transform. Deviation from these is penalized.

        The coefficients appear linearly so we can write
        A x = B, where:
            A   = [[x y 1 0 0 0]
                   [0 0 0 x y 1]]
            Htarget.T = [a0 a1 a2 b0 b1 b2]
            B.T = [X Y]

        In case of ordinary least-squares (OLS) the solution of this system
        of equations is:
        H = np.linalg.inv(A.T @ A) @ A @ B
        
        In case of least-squares with Tikhonov-like regularization:
        H = np.linalg.inv(A.T @ A + Γ.T @ Γ) @ (A @ B + Γ.T @ Γ @ Htarget)
        where Γ.T @ Γ (for simplicity will call it L2 vector) is regularization term and Htarget 
        is a target solution, deviation from which is minimized in L2 sense
     """ 

    src_matrix, src = _center_and_normalize_points_gs(src_pts)
    dst_matrix, dst = _center_and_normalize_points_gs(dst_pts)

    n, d = src.shape
    n2 = n*n   # normalization factor, so that shrinkage parameter does not depend on the number of points

    A = np.zeros((n * d, d * (d + 1)))
    # fill the A matrix with the appropriate block matrices; see docstring
    # for 2D example — this can be generalised to more blocks in the 3D and
    # higher-dimensional cases.
    for ddim in range(d):
        A[ddim*n : (ddim + 1) * n, ddim * (d + 1) : ddim * (d + 1) + d] = src
        A[ddim*n : (ddim + 1) * n, ddim * (d + 1) + d] = 1

    AtA = A.T @ A / n2
    
    if l2_matrix is None:
        l2 = 1.0e-5   # default shrinkage parameter
        l2_matrix = np.eye(2 * (d + 1)) * l2
        for ddim in range(d):
            ii = (d + 1) * (ddim + 1) - 1
            l2_matrix[ii,ii] = 0

    if targ_vector is None:
        targ_vector = np.zeros(2 * (d + 1))
        targ_vector[0] = 1
        targ_vector[4] = 1


    Hp = np.linalg.inv(AtA + l2_matrix) @ (A.T @ dst.T.ravel() / n2 + l2_matrix @ targ_vector)
    Hm = np.eye(d + 1)
    Hm[0:d, 0:d+1] = Hp.reshape(d, d + 1)
    H = np.linalg.inv(dst_matrix) @ Hm @ src_matrix
    return H

def _umeyama(src, dst, estimate_scale):
    """
    Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


class ShiftTransform(ProjectiveTransform):
    """
    Shift transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form:
        X = x +  a2
        Y = y + b2
    and the homogeneous transformation matrix is::
        [[1  0   a2]
         [0   1  b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    """

    def __init__(self, matrix=None, translation=None, *, dimensionality=2):
        
        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if translation is not None and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if translation is not None and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
            self.params[0,1] = 0
            self.params[1,0] = 0
        elif translation is not None:  # note: 2D only
            self.params = np.array([[1.0, 0.0,  0.0],
                                    [0.0,  1.0, 0.0],
                                    [0.0,  0.0, 1.0]])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)
    def estimate(self, src, dst):
        '''
                Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        '''
        translation = np.mean(np.array(dst.astype(float)-src.astype(float)), axis=0)
        self.params = np.array([[1.0, 0.0,  0.0],
                                [0.0,  1.0, 0.0],
                                [0.0,  0.0, 1.0]])
        self.params[0:2, 2] = translation
        return True
        
    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class XScaleShiftTransform(ProjectiveTransform):
    '''
    XScaleShift transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form::
        X = a0*x +  a2 = sx*x + a2
        Y = y + b2 = y + b2
    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::
        [[a0  0   a2]
         [0   1   b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    '''

    def __init__(self, matrix=None, scale=None, translation=None, *, dimensionality=2):
        params = (scale is not None) or (translation is not None)
        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
            self.params[0,1] = 0
            self.params[1,0] = 0
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)

            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = scale
            else:
                sx = scale[0]

            self.params = np.array([[sx, 0,  0],
                                    [0,  1, 0],
                                    [0,  0,  1]])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)
    def estimate(self, src, dst):
        """
                Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
 
        n, d = src.shape
        xsrc = np.array(src)[:,0].astype(float)
        ysrc = np.array(src)[:,1].astype(float)
        xdst = np.array(dst)[:,0].astype(float)
        ydst = np.array(dst)[:,1].astype(float)
        s00 = np.sum(xsrc)
        s01 = np.sum(xdst)
        sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
        #s10 = np.sum(ysrc)
        #s11 = np.sum(ydst)
        #sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)
        sy = 1.0

        tx = np.mean(xdst) - sx * np.mean(xsrc)
        ty = np.mean(ydst) - sy * np.mean(ysrc)

        self.params = np.array([[sx, 0,  tx],
                                [0,  sy, ty],
                                [0,  0,  1]])
        return True
    
    def print_res(self):
        print('Printing from inside the class XScaleShiftTransform')

    @property
    def scale(self):
        return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class ScaleShiftTransform(ProjectiveTransform):
    '''
    ScaleShift transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form::
        X = a0*x +  a2 = sx*x + a2
        Y = b1*y + b2 = sy*y + b2
    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::
        [[a0  0   a2]
         [0   b1  b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    '''

    def __init__(self, matrix=None, scale=None, translation=None, *, dimensionality=2):
        params = (scale is not None) or (translation is not None)
        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
            self.params[0,1] = 0
            self.params[1,0] = 0
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)

            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            self.params = np.array([[sx, 0,  0],
                                    [0,  sy, 0],
                                    [0,  0,  1]])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)
    def estimate(self, src, dst):
        """
                Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
 
        n, d = src.shape
        xsrc = np.array(src)[:,0].astype(float)
        ysrc = np.array(src)[:,1].astype(float)
        xdst = np.array(dst)[:,0].astype(float)
        ydst = np.array(dst)[:,1].astype(float)
        s00 = np.sum(xsrc)
        s01 = np.sum(xdst)
        sx = (n*np.dot(xsrc, xdst) - s00*s01)/(n*np.sum(xsrc*xsrc) - s00*s00)
        s10 = np.sum(ysrc)
        s11 = np.sum(ydst)
        sy = (n*np.dot(ysrc, ydst) - s10*s11)/(n*np.sum(ysrc*ysrc) - s10*s10)

        tx = np.mean(xdst) - sx * np.mean(xsrc)
        ty = np.mean(ydst) - sy * np.mean(ysrc)

        self.params = np.array([[sx, 0,  tx],
                                [0,  sy, ty],
                                [0,  0,  1]])
        return True
        
    @property
    def scale(self):
        return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]


class RotationShiftTransform(ProjectiveTransform):
    """
    Rotation and shift (rigid) transformation. ©G.Shtengel gleb.shtengel@gmail.com

    Has the following form::
        X = cos(θ)*x - sin(θ)*y + tx
        Y = sin(θ)*x + cos(θ)*y + ty
    and the homogeneous transformation matrix is::
        [[cos(θ)  -sin(θ)   tx]
         [sin(θ)   cos(θ)   ty]
         [0        0        1 ]]
    No scale change is applied.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If provided, rotation and translation must be None.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians. Only available for 2D.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. Not used if other parameters are provided.

    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    """

    def __init__(self, matrix=None, rotation=None, translation=None, *, dimensionality=2):
        params = (rotation is not None) or (translation is not None)
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            dimensionality = matrix.shape[0] - 1
            self._coeffs = range(dimensionality * (dimensionality + 1))
            self.params = matrix
        elif params:  # note: 2D only
            if rotation is None:
                rotation = 0
            if translation is None:
                translation = (0, 0)
            self.params = np.array([
                [np.cos(rotation), -np.sin(rotation), 0],
                [np.sin(rotation),  np.cos(rotation), 0],
                [0,                 0,                 1]
            ])
            self.params[0:2, 2] = translation
        else:
            self.params = np.eye(dimensionality + 1)

    def estimate(self, src, dst):
        """
        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        """
        T = _umeyama(src.astype(float), dst.astype(float), estimate_scale=False)
        if np.any(np.isnan(T)):
            return False
        self.params = T
        return True

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The rotation property is only implemented for 2D transforms.'
            )
        return np.arctan2(self.params[1, 0], self.params[0, 0])

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]



class RegularizedAffineTransform(ProjectiveTransform):
    """
    Regularized Affine transformation. ©G.Shtengel 11/2021 gleb.shtengel@gmail.com

    Has the following form::
        X = a0*x + a1*y + a2 =
          = sx*x*cos(rotation) - sy*y*sin(rotation + shear) + a2
        Y = b0*x + b1*y + b2 =
          = sx*x*sin(rotation) + sy*y*cos(rotation + shear) + b2
    where ``sx`` and ``sy`` are scale factors in the x and y directions,
    and the homogeneous transformation matrix is::
        [[a0  a1  a2]
         [b0  b1  b2]
         [0   0    1]]
    In 2D, the transformation parameters can be given as the homogeneous
    transformation matrix, above, or as the implicit parameters, scale,
    rotation, shear, and translation in x (a2) and y (b2). For 3D and higher,
    only the matrix form is allowed.
    In narrower transforms, such as the Euclidean (only rotation and
    translation) or Similarity (rotation, translation, and a global scale
    factor) transforms, it is possible to specify 3D transforms using implicit
    parameters also.

    Parameters
    ----------
    matrix : (D+1, D+1) array, optional
        Homogeneous transformation matrix. If this matrix is provided, it is an
        error to provide any of scale, rotation, shear, or translation.
    scale : {s as float or (sx, sy) as array, list or tuple}, optional
        Scale factor(s). If a single value, it will be assigned to both
        sx and sy. Only available for 2D.
        .. versionadded:: 0.17
           Added support for supplying a single scalar value.
    rotation : float, optional
        Rotation angle in counter-clockwise direction as radians. Only
        available for 2D.
    shear : float, optional
        Shear angle in counter-clockwise direction as radians. Only available
        for 2D.
    translation : (tx, ty) as array, list or tuple, optional
        Translation parameters. Only available for 2D.
    dimensionality : int, optional
        The dimensionality of the transform. This is not used if any other
        parameters are provided.
    Attributes
    ----------
    params : (D+1, D+1) array
        Homogeneous transformation matrix.
    Raises
    ------
    ValueError
        If both ``matrix`` and any of the other parameters are provided.
    """

    def __init__(self, matrix=None, scale=None, rotation=None, shear=None,
                 translation=None, l2_matrix =None, targ_vector=None, *, dimensionality=2):

        self.l2_matrix = l2_matrix      # regularization vector
        self.targ_vector = targ_vector  # target 
        params = any(param is not None
                     for param in (scale, rotation, shear, translation))

        # these parameters get overwritten if a higher-D matrix is given
        self._coeffs = range(dimensionality * (dimensionality + 1))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        if params and dimensionality > 2:
            raise ValueError('Parameter input is only supported in 2D.')
        elif matrix is not None:
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Invalid shape of transformation matrix.")
            else:
                dimensionality = matrix.shape[0] - 1
                nparam = dimensionality * (dimensionality + 1)
            self._coeffs = range(nparam)
            self.params = matrix
        elif params:  # note: 2D only
            if scale is None:
                scale = (1, 1)
            if rotation is None:
                rotation = 0
            if shear is None:
                shear = 0
            if translation is None:
                translation = (0, 0)

            if np.isscalar(scale):
                sx = sy = scale
            else:
                sx, sy = scale

            self.params = np.array([
                [sx * np.cos(rotation), -sy * np.sin(rotation + shear), 0],
                [sx * np.sin(rotation),  sy * np.cos(rotation + shear), 0],
                [                      0,                                0, 1]
            ])
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(dimensionality + 1)


    @property
    def scale(self):
        return np.sqrt(np.sum(self.params ** 2, axis=0))[:self.dimensionality]

    @property
    def rotation(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The rotation property is only implemented for 2D transforms.'
            )
        return np.atan2(self.params[1, 0], self.params[0, 0])

    @property
    def shear(self):
        if self.dimensionality != 2:
            raise NotImplementedError(
                'The shear property is only implemented for 2D transforms.'
            )
        beta = np.atan2(- self.params[0, 1], self.params[1, 1])
        return beta - self.rotation

    @property
    def translation(self):
        return self.params[0:self.dimensionality, self.dimensionality]



# These are functions used for different steps of image analysis and registration

def ShiftTransform0(matrix=None, translation=None):
    return EuclideanTransform(matrix=matrix, rotation = 0, translation = translation)

def ScaleShiftTransform0(matrix=None, scale=None, translation=None):
    return AffineTransform(matrix=matrix, scale=scale, rotation = 0, shear=0, translation = translation)

