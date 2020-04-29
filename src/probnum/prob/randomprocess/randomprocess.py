"""
Random / Stochastic processes.

This module implements classes and functions representing random processes,
i.e. families of random variables.
"""

from abc import ABC, abstractmethod
import numpy as np


class RandomProcess(ABC):
    """
    Random process.

    Random processes are (uncountable) collections of random variables,
    often representing numerical values of some system randomly changing
    over time.


    As a guideline of how to implement which type of random process,
    see below:

    .. csv-table::
        :header: , Instantiation,

        **Discrete time**, "Discrete-time processes are either defined
        through transition densities (``DiscreteProcess(transition)``)"
        **Continuous time**, "Continuous-time processes are defined
        through SDEs (``ContinuousProcess(sde)``)"
        **Space and time**, "``GaussianProcess(meanfun, covfun)``;
            spatiotemporal processes are only defined if they are
            Gaussian processes.",

    The distinction between countable states and continuous states is
    made through the range of the initial random variable (the support
    of its distribution).

    Parameters
    ----------
    """

    def __init__(self, initrv=None, shape=None, dtype=None):
        """Create a new random process."""
        self._initrv = initrv
        self._shape = shape  # todo: check consistency with initrv.shape
        self._dtype = dtype  # todo: check consistency with initrv.dtype

    @abstractmethod
    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        raise NotImplementedError

    @abstractmethod
    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    @abstractmethod
    def covfun(self, x1, x2):
        """
        Covariance (function) of the random process,
        also known as kernel.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, x, size=(), **kwargs):
        """
        Draw realizations from the random process.
        """
        raise NotImplementedError

    @abstractmethod
    def condition(self, start, stop, randvar):
        """
        Conditions the random process on distribution randvar
        at time start. Returns RandomVariable representing its
        distribution at time stop.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, start, stop, value):
        """
        Forwards a particle ``value`` according to the dynamics.
        Returns RandomVariable representing its
        distribution at time stop.

        This function allows using a random process like a transition
        density, sometimes without being one.
        """
        raise NotImplementedError

    @property
    def initrv(self):
        """
        RandomVariable representing the distribution at time :math:`t_0`.
        """
        return self._initrv

    @property
    def range(self):
        """
        Range of the random process. Which values is it going to attend?
        """
        return self._initrv.range

    @property
    def shape(self):
        """Shape of realizations of the random process."""
        return self._shape

    @property
    def dtype(self):
        """Data type of (elements of) a realization of this random process."""
        return self._dtype

    @property
    def random_state(self):
        """Random state of the random process."""
        # todo
        raise NotImplementedError

    @random_state.setter
    def random_state(self, seed):
        """ Get or set the RandomState object of the underlying distribution.

        This can be either None or an existing
        :class:`~numpy.random.RandomState` object. If None
        (or np.random), use the :class:`~numpy.random.RandomState`
        singleton used by np.random. If already a
        :class:`~numpy.random.RandomState` instance, use it. If an int,
        use a new :class:`~numpy.random.RandomState` instance seeded
        with seed.
        """
        # todo
        raise NotImplementedError


class ContinuousProcess(RandomProcess):
    """
    Uniquely defined through solving an SDE with some initial value
    distribution.
    """
    def __init__(self, initrv=None, sde=None):
        """ """
        self._sde = sde
        super().__init__(initrv=initrv, shape=initrv.shape, dtype=initrv.dtype)

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        raise NotImplementedError

    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def covfun(self, x1, x2):
        """
        Covariance (function) of the random process,
        also known as a kernel.
        """
        raise NotImplementedError

    def sample(self, x, size=(), **kwargs):
        """
        Draw realizations from the random process.
        """
        if size == ():
            return self._sample_path(x, **kwargs)
        else:
            return np.array([self._sample_path(x, **kwargs)
                             for __ in range(size)])

    def _sample_path(self, x, **kwargs):
        """
        Draw a realization from the random process.
        """
        raise NotImplementedError

    def condition(self, start, stop, randvar):
        """
        Conditions the random process on distribution randvar
        at time start. Returns RandomVariable representing its
        distribution at time stop.
        """
        # todo: use the sde.
        raise NotImplementedError

    def forward(self, start, stop, value):
        """
        Forwards a particle ``value`` according to the dynamics.
        Returns RandomVariable representing its
        distribution at time stop.

        This function allows using a random process like a transition
        density, sometimes without being one.
        """
        # todo: use the sde.
        raise NotImplementedError

    @property
    def sde(self):
        """
        Stochastic differential equation defining the random process.
        """
        if self._sde is not None:
            return self._sde
        else:
            raise NotImplementedError


class DiscreteProcess(RandomProcess):
    """
    """
    def __init__(self, initrv=None, transition=None):
        """
        dens : Transition
            transition density p(x_i | x_{i-1})
        seq : sequence (array) of RandomVariables
        """
        self._transition = transition
        super().__init__(initrv=initrv, shape=initrv.shape, dtype=initrv.dtype)

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        raise NotImplementedError

    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def covfun(self, x1, x2):
        """
        Covariance (function) of the random process,
        also known as kernel.
        """
        raise NotImplementedError

    def sample(self, x, size=(), **kwargs):
        """
        Draw realizations from the random process.
        """
        raise NotImplementedError

    def _sample_path(self, x, **kwargs):
        """
        Draw a realization from the random process.
        """
        raise NotImplementedError

    def condition(self, start, stop, randvar):
        """
        Conditions the random process on distribution randvar
        at time start. Returns RandomVariable representing its
        distribution at time stop.
        """
        # todo: use the sde.
        raise NotImplementedError

    def forward(self, start, stop, value):
        """
        Forwards a particle ``value`` according to the dynamics.
        Returns RandomVariable representing its
        distribution at time stop.

        This function allows using a random process like a transition
        density, sometimes without being one.
        """
        # todo: use the sde.
        raise NotImplementedError

    @property
    def transition(self):
        """
        Transition density defining the random process.
        """
        return self._transition


class GaussianProcess(RandomProcess):
    """
    We dont subclass from ContinuousProcess but from RandomProcess
    because we will not reuse any SDE machinery here.
    If you like to define a Gauss-Markov process use ContinuousProcess
    with a linear SDE and Gaussian initial variable
    """
    def __init__(self, meanfun=None, covfun=None):
        """
        """
        self._meanfun = meanfun
        self._covfun = covfun

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        raise NotImplementedError

    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def covfun(self, x1, x2):
        """
        Covariance (function) of the random process,
        also known as kernel.
        """
        raise NotImplementedError

    def sample(self, x, size=(), nsteps=1):
        """
        Draw realizations from the random process.
        """
        # todo: use the transition density.
        raise NotImplementedError

    def sample_path(self, x, nsteps=1):
        """
        Draw realizations from the random process.
        """
        raise NotImplementedError

    def condition(self, start, stop, randvar):
        """
        Conditions the random process on distribution randvar
        at time start. Returns RandomVariable representing its
        distribution at time stop.
        """
        # todo: use the sde.
        raise NotImplementedError

    def forward(self, start, stop, value):
        """
        Forwards a particle ``value`` according to the dynamics.
        Returns RandomVariable representing its
        distribution at time stop.

        This function allows using a random process like a transition
        density, sometimes without being one.
        """
        # todo: use the sde.
        raise NotImplementedError
