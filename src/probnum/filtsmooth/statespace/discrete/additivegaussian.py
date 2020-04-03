"""
Discrete models with additive Gaussian noise of the form
x_{i+1} = N(g(i, x_i), S(i))
"""

from diffeq.randomvariable import gaussian
from diffeq.statespace.discretemodel import discretemodel


class Nonlinear(discretemodel.DiscreteModel):
    """
    Discrete Gauss-Markov models of the form
    x_{i+1} = N(g(t_i, x_i), S(t_i)),

    Terminology
    -----------
    g : dynamics
    x : state
    S : diffusion matrix
    """

    def __init__(self, dynafct, diffmatfct, jacfct=None):
        """
        dynafct and jacfct have sign. (t, x, *)
        diffmatfct has sign. (t, *)
        """
        self._dynafct = dynafct
        self._diffmatfct = diffmatfct
        self._jacfct = jacfct

    def dynamics(self, time, state, *args, **kwargs):
        """
        Evaluate g(t_i, x_i).
        """
        dynas = self._dynafct(time, state, *args, **kwargs)
        return dynas

    def jacobian(self, time, state, *args, **kwargs):
        """
        Evaluate Jacobian, d_x g(t_i, x_i),
        of g(t_i, x_i) w.r.t. x_i.
        """
        if self._jacfct is None:
            raise NotImplementedError("Jacobian not provided")
        else:
            return self._jacfct(time, state, *args, **kwargs)

    def diffusionmatrix(self, time, *args, **kwargs):
        """
        Evaluate S(t_i)
        """
        return self._diffmatfct(time, *args, **kwargs)

    def sample(self, time, state, *args, **kwargs):
        """
        Samples x_{t} ~ p(x_{t} | x_{s})
        as a function of t and x_s (plus additional parameters).

        In a discrete system, i.e. t = s + 1, s \\in \\mathbb{N}

        In an ODE solver setting, one of the additional parameters
        would be the step size.
        """
        dynavl = self.dynamics(time, state, *args, **kwargs)
        diffvl = self.diffusionmatrix(time, *args, **kwargs)
        rv = gaussian.MultivariateGaussian(dynavl, diffvl)
        return rv.sample(1)[0, :]

    def pdf(self, loc, time, state, *args, **kwargs):
        """
        Evaluates "future" pdf p(x_t | x_s) at loc.
        """
        dynavl = self.dynamics(time, state, *args, **kwargs)
        diffvl = self.diffusionmatrix(time, *args, **kwargs)
        rv = gaussian.MultivariateGaussian(dynavl, diffvl)
        return rv.pdf(loc)

    @property
    def ndim(self):
        """
        """
        return len(self.diffusionmatrix(0.0))


class Linear(Nonlinear):
    """
    Linear version. g(t, x(t)) = G(t) x(t) + z(t).
    """

    def __init__(self, dynamatfct, forcefct, diffmatfct):
        """
        """

        def dynafct(t, x, *args, **kwargs):
            return dynamatfct(t, *args, **kwargs) @ x + forcefct(t, *args,
                                                                 **kwargs)

        def jacfct(t, x, *args, **kwargs):
            return dynamatfct(t, *args, **kwargs)

        Nonlinear.__init__(self, dynafct, diffmatfct, jacfct)
        self.forcefct = forcefct

    def dynamicsmatrix(self, time, *args, **kwargs):
        """
        Convenient access to dynamics matrix
        (alternative to "jacobian").
        """
        return self.jacobian(time, None, *args, **kwargs)

    def force(self, time, *args, **kwargs):
        """
        """
        return self.forcefct(time, *args, **kwargs)


class LTI(Linear):
    """
    Discrete Gauss-Markov models of the form
    x_{i+1} = N(G x_i + z, S),
    """

    def __init__(self, dynamat, forcevec, diffmat):
        """
        """
        Linear.__init__(self, lambda t: dynamat, lambda t: forcevec,
                        lambda t: diffmat)
