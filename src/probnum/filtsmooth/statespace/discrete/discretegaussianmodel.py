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
    Discrete Gauss-Markov models of the form
    x_{i+1} = N(g(t_i, x_i), S(t_i)),

    Notes
    -----------
    g : dynamics
    x : state
    S : diffusion matrix

    See Also
    --------
    DiscreteGaussianLinearModel :
    DiscreteGaussianLTIModel :
    """

    def __init__(self, dynafct, diffmatfct, jacfct=None):
        """
        dynafct and jacfct have signature (t, x, *)
        diffmatfct has sign. (t, *)
        """
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
        Convenient access to dynamics matrix
        (alternative to "jacobian").
        """
        return self._diffmatfct(time, **kwargs)

    def dynamics(self, time, state, **kwargs):
        return self._dynafct(time, state)

    def jacobian(self, time, state, **kwargs):
        return self._jacfct(time, state)



class DiscreteGaussianLinearModel(DiscreteGaussianModel):
    """
    Linear version. g(t, x(t)) = G(t) x(t) + z(t).
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
        Convenient access to dynamics matrix
        (alternative to "jacobian").
        """
        return self._jacfct(time, None, **kwargs)

    def forcevector(self, time, **kwargs):
        return self._forcefct(time, **kwargs)


class DiscreteGaussianLTIModel(DiscreteGaussianLinearModel):
    """
    Discrete Gauss-Markov models of the form
    x_{i+1} = N(G x_i + z, S),
    """

    def __init__(self, dynamat, forcevec, diffmat):
        super().__init__(
            lambda t, **kwargs: dynamat,
            lambda t, **kwargs: forcevec,
            lambda t, **kwargs: diffmat,
        )

    def transition_realization(self, real, start=None, stop=None):
        return super().transition_realization(real=real, start=None, stop=None)  # no more 'start' necessary

    def transition_rv(self, rv, start=None, stop=None, *args):
        return super().transition_rv(rv=rv, start=None, stop=None)  # no more 'start' necessary
