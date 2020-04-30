"""
Random / Stochastic processes.

This module implements classes and functions representing random processes,
i.e. families of random variables.

Incomplete
----------
* dtype and shape initialisations (could use some help)
* Working with random_state's (could use some help)
* Unittests (matter of time)
* Documentation (matter of time)
"""

import numpy as np





class RandomProcess:
    """
    Parameters
    ----------
    randvars : seq
        Sequence of random variables.
    support : seq
        Support of the sequence of random variables.
    """
    def __init__(self, randvars, support=None):
        """
        Random process as a sequence of random variables.
        """
        self.randvars = randvars
        if support is None:
            self.support = list(range(len(randvars)))
        else:
            assert len(support) == len(randvars)
            self.support = support

    def __call__(self, x):
        """
        Find index of x==self.support and return corresponding random
        variable.
        """
        try:
            return self.randvars[self.support.index(x)]
        except ValueError:
            errormsg = "Random process is not supported at that point"
            raise ValueError(errormsg)

    def __getitem__(self, item):
        """ """
        return self.randvars.__getitem__(item)

    def __len__(self):
        """ """
        return self.randvars.__len__























































class RandomProcess:
    """
    Random process.

    Random processes are (uncountable) collections of random variables,
    often representing numerical values of some system randomly changing
    over time. A more precise term would be "random field"
    but since this choice would rather be confusing than helpful we
    will stick with "random process".


    For a guideline of how to implement which type of random process,
    see below:

    **Markov processes**

    .. csv-table::
        :header: , Instantiation,

        **Discrete time**, "Use a transition density with a support on
        a list of values."
        **Discrete space**, "Use a transition density with a support on
        a graph (graph consists of nodes and edges);
        this is a so-called Markov random field)."
        **Continuous time**, "Use an SDE object as a transition
        density."
        **Continuous space**, "This is only defined for Gaussian
        processes; initialise :class:`GaussianProcess` in this case."

    **Non-Markov processes**

    .. csv-table::
        :header: , Instantiation,

        **Discrete time**, "What is a discrete-time non-Markov process?"
        **Discrete space**, "What is a discrete-space non-Markov process?"
        **Continuous time**, "Use a corresponding SDE object as a
        transition density."
        **Continuous space**, "Gaussian processes are usually not
        Markovian by default."


    The distinction between countable states and continuous states is
    made through the range of the initial random variable (the support
    of its distribution).

    Parameters
    ----------
    """

    def __init__(self, transition=None, initrv=None, shape=None,
                 dtype=None):
        """Create a new random process."""
        self._initrv = initrv
        self._shape = shape  # todo: check consistency with initrv.shape
        self._dtype = dtype  # todo: check consistency with initrv.dtype
        self._transition = transition

    def __call__(self, x):
        """
        Returns random variable corresponding to the random process
        evaluated at point ``x``.
        """
        raise NotImplementedError

    def meanfun(self, x):
        """
        Evaluates mean (function) of the random process at :math:`x`.
        """
        raise NotImplementedError

    def covfun(self, x1, x2):
        """
        Evaluates covariance (function; also known as kernel)
        of the random process at :math:`x`.
        """
        raise NotImplementedError

    def sample(self, size=(), x=None, **kwargs):
        """
        Draw realizations from the random process.

        Parameters
        ----------
        x : array_like, optional.
            If None, the full trajectory is sampled.
            If array_like, the values of the random process at
            times x are sampled.
        size : tuple, optional.
            How many samples?
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
        if x is None:
            return self._transition.sample(**kwargs)
        else:
            return self.__call__(x).sample()

    def condition(self, start, stop, randvar, **kwargs):
        """
        Conditions the random process on randvar at start.

        If randvar has an explicit parameterization (e.g. mean and
        covariance for a Normal distribution) and this can be propagated
        through the transition, this function returns the
        parameterization of the propagated distribution.

        In the SDE setting, this only works if there is a closed form
        solution to the SDEs. In the discrete setting,

        Returns RandomVariable representing its
        distribution at time stop.
        """
        if self._transition is None:
            raise NotImplementedError
        else:
            return self._transition.condition(start, stop, randvar, **kwargs)

    def forward(self, start, stop, value, **kwargs):
        """
        Forwards a single particle according to the dynamics.

        Returns RandomVariable representing its
        distribution at time stop.

        This function allows using a random process like a transition
        density.
        """
        if self._transition is None:
            raise NotImplementedError
        else:
            return self._transition.forward(start, stop, value, **kwargs)

    @property
    def transition(self):
        """
        Returns Transition object that defines the random process.

        If the process is discrete, this will be a direct subclass of
        :class:`Transition`. If the process is continuous, it will be
        a subclass of :class:`SDE`.
        """
        return self._transition

    @property
    def initrv(self):
        """
        RandomVariable representing the distribution at time :math:`t_0`.
        """
        return self._initrv

    @property
    def support(self):
        """
        """
        return self._transition.support

    @support.setter
    def support(self, newsupport):
        """
        """
        self._transition.support(newsupport)

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
