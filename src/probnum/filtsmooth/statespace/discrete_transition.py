"""Discrete transitions."""
from typing import Callable, Optional

import numpy as np

import probnum.random_variables as pnrv
from probnum.type import FloatArgType

from . import transition as trans


class DiscreteGaussian(trans.Transition):
    """Random variable transitions with additive Gaussian noise.

    .. math:: x_{i+1} \\sim \\mathcal{N}(g(t_i, x_i), S(t_i))

    for some (potentially non-linear) dynamics :math:`g` and diffusion matrix :math:`S`.
    This is used for, but not restricted to time-series.

    Parameters
    ----------
    dynamicsfun :
        Dynamics function :math:`g=g(t, x)`. Signature: ``dynafct(t, x)``.
    diffmatfun :
        Diffusion matrix function :math:`S=S(t)`. Signature: ``diffmatfct(t)``.
    jacobfun :
        Jacobian of the dynamics function :math:`g`, :math:`Jg=Jg(t, x)`.
        Signature: ``jacfct(t, x)``.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self,
        dynamicsfun: Callable[[FloatArgType, np.ndarray], np.ndarray],
        diffmatfun: Callable[[FloatArgType], np.ndarray],
        jacobfun: Optional[Callable[[FloatArgType, np.ndarray], np.ndarray]] = None,
    ):
        self.dynamicsfun = dynamicsfun
        self.diffmatfun = diffmatfun

        def if_no_jacobian(t, x):
            raise NotImplementedError

        self.jacobfun = jacobfun if jacobfun is not None else if_no_jacobian
        super().__init__()

    def transition_realization(self, real, start, **kwargs):

        newmean = self.dynamicsfun(start, real)
        newcov = self.diffmatfun(start)
        crosscov = np.zeros(newcov.shape)
        return pnrv.Normal(newmean, newcov), {"crosscov": crosscov}

    def transition_rv(self, rv, start, **kwargs):

        raise NotImplementedError


class DiscreteLinearGaussian(DiscreteGaussian):
    """Discrete, linear Gaussian transition models of the form.

    .. math:: x_{i+1} \\sim \\mathcal{N}(G(t_i) x_i + v(t_i), S(t_i))

    for some dynamics matrix :math:`G=G(t)`, force vector :math:`v=v(t)`,
    and diffusion matrix :math:`S=S(t)`.


    Parameters
    ----------
    dynamatfct : callable
        Dynamics function :math:`G=G(t)`. Signature: ``dynamatfct(t)``.
    forcefct : callable
        Force function :math:`v=v(t)`. Signature: ``forcefct(t)``.
    diffmatfct : callable
        Diffusion matrix function :math:`S=S(t)`. Signature: ``diffmatfct(t)``.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self,
        dynamicsmatfun: Callable[[FloatArgType], np.ndarray],
        forcevecfun: Callable[[FloatArgType], np.ndarray],
        diffmatfun: Callable[[FloatArgType], np.ndarray],
    ):

        self.dynamicsmatfun = dynamicsmatfun
        self.forcevecfun = forcevecfun

        super().__init__(
            dynamicsfun=lambda t, x: (self.dynamicsmatfun(t) @ x + self.forcevecfun(t)),
            diffmatfun=diffmatfun,
            jacobfun=lambda t, x: dynamicsmatfun(t),
        )

    def transition_rv(self, rv, start, **kwargs):

        if not isinstance(rv, pnrv.Normal):
            raise TypeError(f"Normal RV expected, but {type(rv)} received.")

        dynamicsmat = self.dynamicsmatfun(start)
        diffmat = self.diffmatfun(start)
        force = self.forcevecfun(start)

        new_mean = dynamicsmat @ rv.mean + force
        new_crosscov = rv.cov @ dynamicsmat.T
        new_cov = dynamicsmat @ new_crosscov + diffmat
        return pnrv.Normal(mean=new_mean, cov=new_cov), {"crosscov": new_crosscov}

    @property
    def dimension(self):
        # risky to evaluate at zero, but works well
        # remove this -- the dimension of discrete transitions is not clear!
        # input dim != output dim is possible...
        # See issue #266
        return len(self.dynamicsmatfun(0.0).T)


class DiscreteLTIGaussian(DiscreteLinearGaussian):
    """Discrete, linear, time-invariant Gaussian transition models of the form.

    .. math:: x_{i+1} \\sim \\mathcal{N}(G x_i + v, S)

    for some dynamics matrix :math:`G`, force vector :math:`v`,
    and diffusion matrix :math:`S`.

    Parameters
    ----------
    dynamat :
        Dynamics matrix :math:`G`.
    forcevec :
        Force vector :math:`v`.
    diffmat :
        Diffusion matrix :math:`S`.

    Raises
    ------
    TypeError
        If dynamat, forcevec and diffmat have incompatible shapes.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(
        self, dynamicsmat: np.ndarray, forcevec: np.ndarray, diffmat: np.ndarray
    ):
        _check_dimensions(dynamicsmat, forcevec, diffmat)

        super().__init__(
            lambda t: dynamicsmat,
            lambda t: forcevec,
            lambda t: diffmat,
        )

        self.dynamicsmat = dynamicsmat
        self.forcevec = forcevec
        self.diffmat = diffmat


def _check_dimensions(dynamicsmat, forcevec, diffmat):
    if dynamicsmat.ndim != 2:
        raise TypeError(
            f"dynamat.ndim=2 expected. dynamat.ndim={dynamicsmat.ndim} received."
        )
    if forcevec.ndim != 1:
        raise TypeError(
            f"forcevec.ndim=1 expected. forcevec.ndim={forcevec.ndim} received."
        )
    if diffmat.ndim != 2:
        raise TypeError(
            f"diffmat.ndim=2 expected. diffmat.ndim={diffmat.ndim} received."
        )
    if (
        dynamicsmat.shape[0] != forcevec.shape[0]
        or forcevec.shape[0] != diffmat.shape[0]
        or diffmat.shape[0] != diffmat.shape[1]
    ):
        raise TypeError(
            f"Dimension of dynamat, forcevec and diffmat do not align. "
            f"Expected: dynamat.shape=(N,*), forcevec.shape=(N,), diffmat.shape=(N, N).     "
            f"Received: dynamat.shape={dynamicsmat.shape}, forcevec.shape={forcevec.shape}, "
            f"diffmat.shape={diffmat.shape}."
        )
