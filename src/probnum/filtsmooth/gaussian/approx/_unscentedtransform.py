"""Unscented Transform."""


import numpy as np


class UnscentedTransform:
    """Used for unscented Kalman filter.

    See also p. 7 ("Unscented transform:") of [1]_.

    Parameters
    ----------
    dimension : int
        Spatial dimensionality
    spread : float
        Spread of the sigma points around mean
    priorpar : float
        Incorporate prior knowledge about distribution of x.
        For Gaussians, 2.0 is optimal (see link below)
    special_scale : float
        Secondary scaling parameter.
        The primary parameter is computed below.

    References
    ----------
    .. [1] Wan, E. A. and van der Merwe, R., The Unscented Kalman Filter,
       http://read.pudn.com/downloads135/ebook/574389/wan01unscented.pdf
    """

    def __init__(self, dimension, spread=1e-4, priorpar=2.0, special_scale=0.0):
        self.scale = _compute_scale(dimension, spread, special_scale)
        self.dimension = dimension
        self.mweights, self.cweights = _unscented_weights(
            spread, priorpar, self.dimension, self.scale
        )

    def sigma_points(self, rv):
        """Sigma points.

        Parameters
        ----------
        mean: np.ndarray, shape (d,)
            mean of Gaussian distribution
        covar: np.ndarray, shape (d, d)
            kernels of Gaussian distribution

        Returns
        -------
        np.ndarray, shape (2 * d + 1, d)
        """
        if len(rv.mean) != self.dimension:
            raise ValueError("Dimensionality does not match UT")
        sigpts = np.zeros((2 * self.dimension + 1, self.dimension))
        sqrtcovar = rv.cov_cholesky
        sigpts[0] = rv.mean.copy()
        for idx in range(self.dimension):
            sigpts[idx + 1] = (
                rv.mean + np.sqrt(self.dimension + self.scale) * sqrtcovar[:, idx]
            )
            sigpts[self.dimension + 1 + idx] = (
                rv.mean - np.sqrt(self.dimension + self.scale) * sqrtcovar[:, idx]
            )
        return sigpts

    def propagate(self, time, sigmapts, modelfct):
        """Propagate sigma points.

        Parameters
        ----------
        time : float
            Time :math:`t` which is passed on to the modelfunction.
        sigmapts : np.ndarray, shape=(2 N+1, N)
            Sigma points (N is the spatial dimension of the dynamic model)
        modelfct : callable, signature=(t, x, \\**kwargs)
            Function through which to propagate

        Returns
        -------
        np.ndarray, shape=(2 N + 1, M),
            M is the dimension of the measurement model
        """
        propsigpts = np.array([modelfct(time, pt) for pt in sigmapts])
        return propsigpts

    def estimate_statistics(self, proppts, sigpts, covmat, mpred):
        """Computes predicted summary statistics, predicted
        mean/kernels/crosscovariance, from (propagated) sigmapoints.

        Not to be confused with mean and kernels resulting from the
        prediction step of the Bayesian filter. Hence we call it
        "estimate_*" instead of "predict_*".
        """
        estmean = _estimate_mean(self.mweights, proppts)
        estcovar = _estimate_covar(self.cweights, proppts, estmean, covmat)
        estcrosscovar = _estimate_crosscovar(
            self.cweights, proppts, estmean, sigpts, mpred
        )
        return estmean, estcovar, estcrosscovar


def _compute_scale(dimension, spread, special_scale):
    """See BFaS; p. 83.

    Parameters
    ----------
    dimension: int
        Spatial dimensionality of state space model
    spread: float
        Spread of sigma points around mean (1; alpha)
    special_scale: float
        Spread of sigma points around mean (2; kappa)

    Returns
    -------
    float
        Scaling parameter for unscented transform
    """
    return spread ** 2 * (dimension + special_scale) - dimension


def _unscented_weights(spread, priorpar, dimension, scale):
    """See BFaS; p. 84.

    Parameters
    ----------
    spread: float
        Spread of sigma points around mean (alpha)
    priorpar: float
        Prior information parameter (beta)
    dimension : int
        Dimension of the state space
    scale : float
        Scaling parameter for unscented transform

    Returns
    -------
    np.ndarray, shape (2 * dimension + 1,)
        constant mean weights.
    np.ndarray, shape (2 * dimension + 1,)
        constant kernels weights.
    """
    mweights = _meanweights(dimension, scale)
    cweights = _covarweights(dimension, spread, priorpar, scale)
    return mweights, cweights


def _meanweights(dimension, lam):
    """Mean weights.

    Parameters
    ----------
    dimension: int
        Spatial dimensionality of state space model
    lam: float
        Scaling parameter for unscented transform (lambda)

    Returns
    -------
    np.ndarray, shape (2*dimension+1,)
        Constant mean weights.
    """
    mw0 = np.ones(1) * lam / (dimension + lam)
    mw = np.ones(2 * dimension) / (2.0 * (dimension + lam))
    return np.hstack((mw0, mw))


def _covarweights(dimension, alp, bet, lam):
    """Covariance weights.

    Parameters
    ----------
    dimension: int
        Spatial dimensionality of state space model
    alp: float
        Spread of sigma points around mean (alpha)
    bet: float
        Prior information parameter (beta)
    lam: float
        Scaling parameter for unscented transform (lambda)

    Returns
    -------
    np.ndarray, shape (2 * dimension + 1,)
        the constant kernels weights.
    """
    cw0 = np.ones(1) * lam / (dimension + lam) + (1 - alp ** 2 + bet)
    cw = np.ones(2 * dimension) / (2.0 * (dimension + lam))
    return np.hstack((cw0, cw))


def _estimate_mean(mweights, proppts):
    """See BFaS; p. 88.

    Parameters
    ----------
    mweights: np.ndarray, shape (2*dimension + 1,)
        Constant mean weights for unscented transform.
    proppts: np.ndarray, shape (2*dimension + 1, dimension)
        Propagated sigma points

    Returns
    -------
    np.ndarray, shape (dimension,)
        Estimated mean.
    """
    return mweights @ proppts


def _estimate_covar(cweights, proppts, mean, covmat):
    """See BFaS; p. 88.

    Parameters
    ----------
    cweights: np.ndarray, shape (2*dimension + 1,)
        Constant kernels weights for unscented transform.
    proppts: np.ndarray, shape (2*dimension + 1, dimension)
        Propagated sigma points
    mean: np.ndarray, shape (dimension,)
        Result of _estimate_mean(...)
    covmat: np.ndarray, shape (dimension, dimension)
        Covariance of measurement model at current time.

    Returns
    -------
    np.ndarray, shape (dimension, dimension)
        Estimated kernels.
    """
    cent = proppts - mean
    empcov = cent.T @ (cweights * cent.T).T
    return empcov + covmat


def _estimate_crosscovar(cweights, proppts, mean, sigpts, mpred):
    """See BFaS; p.88.

    Parameters
    ----------
    cweights: np.ndarray, shape (2*dimension + 1,)
        Constant kernels weights for unscented transform.
    sigpts: np.ndarray, shape (2*dimension + 1, dimension)
        Sigma points
    mpred: np.ndarray, shape (dimension,)
        Predicted mean
    proppts: np.ndarray, shape (2*dimension + 1, dimension)
        Propagated sigma points
    mean: np.ndarray, shape (dimension,)
        Result of _estimate_mean(...)

    Returns
    -------
    np.ndarray, shape (dimension,)
        Estimated kernels.
    """
    cent_prop = proppts - mean
    cent_sig = sigpts - mpred
    empcrosscov = cent_sig.T @ (cweights * cent_prop.T).T
    return empcrosscov
