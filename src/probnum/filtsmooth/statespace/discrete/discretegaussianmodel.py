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
        dynafct and jacfct have sign. (t, x, *)
        diffmatfct has sign. (t, *)
        """
        self._dynafct = dynafct
        self._diffmatfct = diffmatfct
        self._jacfct = jacfct

    def dynamics(self, time, state, **kwargs):
        """
        Evaluate g(t_i, x_i).
        """
        dynas = self._dynafct(time, state, **kwargs)
        return dynas

    def jacobian(self, time, state, **kwargs):
        """
        Evaluate Jacobian, d_x g(t_i, x_i),
        of g(t_i, x_i) w.r.t. x_i.
        """
        if self._jacfct is None:
            raise NotImplementedError("Jacobian not provided")
        else:
            return self._jacfct(time, state, **kwargs)

    def diffusionmatrix(self, time, **kwargs):
        """
        Evaluate S(t_i)
        """
        return self._diffmatfct(time, **kwargs)

    def sample(self, time, state, **kwargs):
        """
        Samples x_{t} ~ p(x_{t} | x_{s})
        as a function of t and x_s (plus additional parameters).

        In a discrete system, i.e. t = s + 1, s \\in \\mathbb{N}

        In an ODE solver setting, one of the additional parameters
        would be the step size.
        """
        dynavl = self.dynamics(time, state, **kwargs)
        diffvl = self.diffusionmatrix(time, **kwargs)
        rv = Normal(dynavl, diffvl)
        return rv.sample()

    def pdf(self, loc, time, state, **kwargs):
        """
        Evaluates "future" pdf p(x_t | x_s) at loc.
        """
        dynavl = self.dynamics(time, state, **kwargs)
        diffvl = self.diffusionmatrix(time, **kwargs)
        normaldist = Normal(dynavl, diffvl)
        return normaldist.pdf(loc)

    @property
    def ndim(self):
        return len(self.diffusionmatrix(0.0))


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
        self.forcefct = forcefct

    def dynamicsmatrix(self, time, **kwargs):
        """
        Convenient access to dynamics matrix
        (alternative to "jacobian").
        """
        return self.jacobian(time, None, **kwargs)

    def force(self, time, **kwargs):
        return self.forcefct(time, **kwargs)


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
