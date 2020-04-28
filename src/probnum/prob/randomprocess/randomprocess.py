"""
Random / Stochastic processes.

This module implements classes and functions representing random processes,
i.e. families of random variables.
"""


class RandomProcess:
    """
    Random process.

    Random processes are collections of random variables, often representing
    numerical values of some system randomly changing over time.

    Parameters
    ----------
    """

    def __init__(self, shape=None, dtype=None, sde=None):
        """Create a new random process."""
        self._shape = shape
        self._dtype = dtype
        self._sde = sde

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process evaluated
        at ``x``.
        """
        raise NotImplementedError

    def meanfun(self, x):
        """
        Mean (function) of the random process.
        """
        raise NotImplementedError

    def covfun(self, x1, x2):
        """
        Covariance (function) of the random process, sometimes known as kernel.
        """
        raise NotImplementedError

    def sde(self):
        """
        Stochastic differential equation defining the random process.
        """
        return self._sde

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
        raise NotImplementedError

    @random_state.setter
    def random_state(self, seed):
        """ Get or set the RandomState object of the underlying distribution.

        This can be either None or an existing :class:`~numpy.random.RandomState` object.
        If None (or np.random), use the :class:`~numpy.random.RandomState` singleton used by np.random.
        If already a :class:`~numpy.random.RandomState` instance, use it.
        If an int, use a new :class:`~numpy.random.RandomState` instance seeded with seed.
        """
        raise NotImplementedError

    def sample(self, x, size=()):
        """
        Draw realizations from the random process.

        Parameters
        ----------
        size : tuple
            Size of the drawn sample of realizations.

        Returns
        -------
        sample : array-like
            Sample of realizations with the given ``size`` and the inherent ``shape``.
        """
        raise NotImplementedError


def asrandprocess():
    """
    """
