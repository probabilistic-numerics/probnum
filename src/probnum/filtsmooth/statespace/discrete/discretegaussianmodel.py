"""
Discrete Gauss-Markov models of the form
x_{i+1} = N(g(i, x_i), S(i))
"""

from probnum.random_variables import Normal
from probnum.filtsmooth.statespace.discrete import discretemodel


__all__ = [
    "DiscreteGaussianModel",
    "DiscreteGaussianLinearModel",
    "DiscreteGaussianLTIModel",
]


class DiscreteGaussianModel(discretemodel.DiscreteModel):
    """
    Discrete Gaussian transition models of the form

    .. math:: x_{i+1} \\sim \\mathcal{N}(g(t_i, x_i), S(t_i))

    for some (potentially non-linear) dynamics :math:`g` and diffusion matrix :math:`S`.


    Parameters
    ----------
    dynafct : callable
        Dynamics function :math:`g=g(t, x)`. Signature: ``dynafct(t, x)``.
    diffmatfct : callable
        Diffusion matrix function :math:`S=S(t)`. Signature: ``diffmatfct(t)``.
    jacfct : callable, optional.
        Jacobian of the dynamics function :math:`g`, :math:`Jg=Jg(t, x)`.
        Signature: ``jacfct(t, x)``.

    See Also
    --------
    :class:`DiscreteModel`
    :class:`DiscreteGaussianLinearModel`
    """

    def __init__(self, dynafct, diffmatfct, jacfct=None):
        self._dynafct = dynafct
        self._diffmatfct = diffmatfct
        self._jacfct = jacfct

    def transition_realization(self, real, start, stop=None):
        newmean = self._dynafct(start, real)
        newcov = self._diffmatfct(start)
        return Normal(newmean, newcov), {}

    def transition_rv(self, rv, start, stop=None, *args):
        raise NotImplementedError

    @property
    def dimension(self):
        return len(self.diffusionmatrix(0.0))

    def diffusionmatrix(self, time, **kwargs):
        """
        Compute diffusion matrix :math:`S=S(t)` at time :math:`t`.

        Parameters
        ----------
        time : float
            Time :math:`t`.

        Returns
        -------
        :class:`np.ndarray`
            Diffusion matrix :math:`S=S(t)`.
        """
        return self._diffmatfct(time, **kwargs)

    def dynamics(self, time, state, **kwargs):
        """
        Compute dynamics :math:`g=g(t, x)` at time :math:`t`
        and state :math:`x`.

        Parameters
        ----------
        time : float
            Time :math:`t`.
        state : array_like
            State :math:`x`. For instance, realization of a random variable.

        Returns
        -------
        :class:`np.ndarray`
            Evaluation of :math:`g=g(t, x)`.
        """
        return self._dynafct(time, state)

    def jacobian(self, time, state, **kwargs):
        """
        Compute diffusion matrix :math:`S=S(t)` at time :math:`t`.

        Parameters
        ----------
        time : float
            Time :math:`t`.
        state : array_like
            State :math:`x`. For instance, realization of a random variable.

        Raises
        ------
        NotImplementedError
            If the Jacobian is not implemented.
            This is the case if :meth:`jacfct` is not specified at initialization.

        Returns
        -------
        :class:`np.ndarray`
            Evaluation of the Jacobian :math:`J g=Jg(t, x)`.
        """
        if self._jacfct is None:
            raise NotImplementedError
        return self._jacfct(time, state)


class DiscreteGaussianLinearModel(DiscreteGaussianModel):
    """
    Discrete, linear Gaussian transition models of the form

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

    def __init__(self, dynamatfct, forcefct, diffmatfct):
        def dynafct(t, x, **kwargs):
            return dynamatfct(t, **kwargs) @ x + forcefct(t, **kwargs)

        def jacfct(t, x, **kwargs):
            return dynamatfct(t, **kwargs)

        super().__init__(dynafct, diffmatfct, jacfct)
        self._forcefct = forcefct

    def transition_rv(self, rv, start, stop=None, *args):
        if not isinstance(rv, Normal):
            raise TypeError(f"Normal RV expected, but {type(rv)} received.")
        dynamat = self.dynamicsmatrix(time=start)
        diffmat = self.diffusionmatrix(time=start)
        force = self.forcevector(time=start)

        new_mean = dynamat @ rv.mean + force
        new_crosscov = rv.cov @ dynamat.T
        new_cov = dynamat @ new_crosscov + diffmat
        return Normal(mean=new_mean, cov=new_cov), {"crosscov": new_crosscov}

    def dynamicsmatrix(self, time, **kwargs):
        """
        Compute dynamics matrix :math:`G=G(t)` at time :math:`t`.
        The output is equivalent to :meth:`jacobian`.

        Parameters
        ----------
        time : float
            Time :math:`t`.

        Returns
        -------
        :class:`np.ndarray`
            Evaluation of the dynamics matrix :math:`G=G(t)`.
        """
        return self._jacfct(time, None, **kwargs)

    def forcevector(self, time, **kwargs):
        """
        Compute force vector :math:`v=v(t)` at time :math:`t`.

        Parameters
        ----------
        time : float
            Time :math:`t`.

        Returns
        -------
        :class:`np.ndarray`
            Evaluation of the force :math:`v=v(t)`.
        """
        return self._forcefct(time, **kwargs)


class DiscreteGaussianLTIModel(DiscreteGaussianLinearModel):
    """
    Discrete, linear, time-invariant Gaussian transition models of the form

    .. math:: x_{i+1} \\sim \\mathcal{N}(G x_i + v, S)

    for some dynamics matrix :math:`G`, force vector :math:`v`,
    and diffusion matrix :math:`S`.

    Parameters
    ----------
    dynamat : np.ndarray
        Dynamics matrix :math:`G`.
    forcevec : np.ndarray
        Force vector :math:`v`.
    diffmat : np.ndarray
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

    def __init__(self, dynamat, forcevec, diffmat):
        if dynamat.ndim != 2 or forcevec.ndim != 1 or diffmat.ndim != 2:
            raise TypeError
        if not dynamat.shape[0] == forcevec.shape[0] == diffmat.shape[0] == diffmat.shape[1]:
            raise TypeError

        super().__init__(
            lambda t, **kwargs: dynamat,
            lambda t, **kwargs: forcevec,
            lambda t, **kwargs: diffmat,
        )

    def transition_realization(self, real, start=None, stop=None):
        return super().transition_realization(
            real=real, start=None, stop=None
        )  # no more 'start' necessary

    def transition_rv(self, rv, start=None, stop=None, *args):
        return super().transition_rv(
            rv=rv, start=None, stop=None
        )  # no more 'start' necessary
