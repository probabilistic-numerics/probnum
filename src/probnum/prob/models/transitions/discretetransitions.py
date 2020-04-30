"""

"""

import numpy as np
from probnum.prob.models.transitions.transitions import Transition
from probnum.prob import RandomVariable, Normal


class GaussianTransition(Transition):
    """
    Transition models with additive Gaussian noise.

    That is, models of the form

    .. math:: p(x_i | x_{i-1}) = N(f(t_{i-1}), x_{i-1}), Q(t_{i-1}),

    are implemented. Jacobian is w.r.t. x.

    Examples
    --------
    >>> from probnum.prob.randomprocess import GaussianTransition
    >>> lgt = GaussianTransition(transfun=(lambda t, x: np.sin(x)),
    ...                                    covfun=(lambda t: 0.1))
    >>> forw = lgt.forward(0, 1, value=0.2)
    >>> print(forw.mean(), forw.cov())
    0.19866933079506122 0.1
    """
    def __init__(self, transfun, covfun, support=None, jacobfun=None):
        """ """
        self._transfun = transfun
        self._covfun = covfun
        self._jacobfun = jacobfun
        super().__init__(support=support)

    # parameter "stop" is only here bc. of the general signature.
    def forward(self, start, stop, value, **kwargs):
        """
        """
        mean = self._transfun(start, value)
        cov = self._covfun(start)
        return RandomVariable(distribution=Normal(mean, cov))

    def condition(self, start, stop, randvar, **kwargs):
        """
        Only works if f=f(t, x) is linear in x.

        See :class:`LinearGaussianTransition`.
        """
        raise NotImplementedError

    def jacobfun(self, t, x):
        """ """
        return self._jacobfun(t, x)


class LinearGaussianTransition(GaussianTransition):
    """
    Linear Gaussian transitions.

    That is, the dynamic transition function :math:`f` is of the form
    :math:`f(t, x) = F(t) x`. This enables conditioning the
    distribution on a previous Gaussian distribution.

    Examples
    --------
    >>> from probnum.prob.randomprocess import LinearGaussianTransition
    >>> lgt = LinearGaussianTransition(lintransfun=(lambda t: 2),
    ...                                covfun=(lambda t: 0.1))
    >>> forw = lgt.forward(0, 1, value=0.2)
    >>> cond = lgt.condition(1, 2, randvar=forw)
    >>> print(forw.mean(), forw.cov())
    0.4 0.1
    >>> print(cond.mean(), cond.cov())
    0.8 0.4
    """
    def __init__(self, lintransfun, covfun, support=None):
        self._lintransfun = lintransfun
        super().__init__(transfun=(lambda t, x: np.dot(lintransfun(t), x)),
                         covfun=covfun, support=support, jacobfun=lintransfun)

    def condition(self, start, stop, randvar, **kwargs):
        """ """
        if not issubclass(type(randvar.distribution), Normal):
            raise ValueError("Input distribution must be a Normal.")
        oldmean, oldcov = randvar.mean(), randvar.cov()
        lintrans = self._lintransfun(start)
        if np.isscalar(oldmean):
            newmean = lintrans * oldmean
            newcov = lintrans**2 * oldcov
        else:
            newmean = lintrans @ oldmean
            newcov = lintrans @ oldcov @ lintrans.T
        return RandomVariable(distribution=Normal(newmean, newcov))

    def lintransfun(self, t):
        """ """
        return self._lintransfun(t)
