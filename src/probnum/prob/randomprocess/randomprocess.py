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

from abc import ABC, abstractmethod
import numpy as np


class _AbstractRandomProcess(ABC):
    """
    Abstract interface for random processes.

    Notes
    -----
    Made abstract because generic RandomProcess instance
    should serve as data structures for lists of RandomVariables
    and as such implement things like getitem, etc.
    GaussianProcesses should not behave like containers,
    the options should not even appear anywhere near their
    documentation.
    """

    def __init__(self, bounds, dtype):
        """ """
        self._bounds = bounds
        self._dtype = dtype

    # Abstract methods: statistics functions ###########################

    @abstractmethod
    def __call__(self, x):
        """ """
        raise NotImplementedError

    @abstractmethod
    def meanfun(self, x):
        """ """
        raise NotImplementedError

    @abstractmethod
    def covfun(self, x):
        """ """
        raise NotImplementedError

    @abstractmethod
    def sample(self, x, size=()):
        """ """
        raise NotImplementedError

    # Properties, getters and setters ##################################

    @property
    def bounds(self):
        """
        Bounds of the support of the random process.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """
        Bounds of the support of the random process.
        """
        if len(self._bounds) != len(bounds):  # incomplete check!!
            errormsg = "Size of bounds does not fit RandomProcess."
            raise ValueError(errormsg)
        self._bounds = bounds

    @property
    def dtype(self):
        """
        Data type of the realizations of the random process.
        """
        raise NotImplementedError("TODO")

    @dtype.setter
    def dtype(self, dtype):
        """
        Data type of the realizations of the random process.
        """
        raise NotImplementedError("TODO")


class RandomProcess(_AbstractRandomProcess):
    """
    Random processes are the in- and output of probabilistic
    numerical algorithms involving time- and space-components.

    A random process is a collection of random variables
    :math:`X = \\{X_{\\tau} \\}_{\\tau \\in \\mathcal{T}}`
    indexed by some topological space :math:`\\mathcal{T}`. More precisely, a
    random process (random field) can be thought of as a map

    .. math:: X : \\mathcal{T} \\longrightarrow RV, \\quad \\tau \\longmapsto X_{\\tau}

    from an element in a topological space to a random variable.
    For this reason the RandomProcess interface behaves like a function.

    The object assumes to be one-dimensional unless either
    ``support`` or ``bounds`` say otherwise.
    Any variable that is ignored is set to ``None``.

    - *Discrete:* As soon as ``support`` is specified, the object assumes to be
      discrete and ``bounds`` is ignored.

    - *Continuous:* If no ``support`` is specified, the object assumes to be continuous.
      If ``bounds`` is not specified either, ``bounds`` are set
      to ``bounds=(-inf, inf)``.


    A continuous time RandomProcess behaves like a callable and a
    numeric type. A discrete time RandomProcess additionally emulates
    container types supporting ``__len__``, ``__getitem__``, etc..


    Parameters
    ----------
    randvars : array_like or callable
        Collection of variables. Either defined as an array
        ``[rv1, rv2, ..., rvN]`` or as a map :math:`\\tau \\rightarrow X_\\tau`.
    support : array_like, optional.
        Support points of the random process.
        Expects shape (len(rvmap), ndim) if rvmap is a sequence.
    bounds : array_like, optional.
        Bounds of the random process.
        Expects shape (ndim, 2), respectively (2,) if ndim is 1.
        Lower and upper bounds of the support for each dimension.

    Raises
    ------
    ValueError
        If dimensions of bounds and support do not match.

    Examples
    --------
    Initialise a ``RandomProcess`` with a finite collection of
    random variables. Here, we use a list of ``RandomVariable`` objects
    with ``Normal`` distribution. We initialize two random processes,
    one with explicit support and one without explicit support.

    >>> import numpy as np
    >>> from probnum.prob.randomprocess import RandomProcess
    >>> from probnum.prob import RandomVariable, Normal
    >>> rvs = [RandomVariable(distribution=Normal(0.0, idx**2))
    ...        for idx in range(20)]
    >>> rp1 = RandomProcess(rvs)
    >>> supp = np.arange(start=0, stop=2*len(rp1), step=2.0)
    >>> rp2 = RandomProcess(rvs, support=supp)

    Both ``RandomProcess`` instances behaves like sequences.

    >>> print(len(rp1))
    20
    >>> print(rp1[2])
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp2[3:7:2])
    [<() RandomVariable with dtype=<class 'float'>>, <() RandomVariable with dtype=<class 'float'>>]
    >>> for el in rp1[:3]:
    ...     print(el)
    <() RandomVariable with dtype=<class 'float'>>
    <() RandomVariable with dtype=<class 'float'>>
    <() RandomVariable with dtype=<class 'float'>>

    They also behave like callables.
    If called directly at input `x`, they return the random variable
    representing the process at time `x`.
    Note that for discrete support points, slicing can be more
    stable than evaluating due to round-off errors.

    >>> print(rp1(2.0))
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp1(2.0).cov())
    4.0
    >>> print(rp2(10.0))
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp1(2.0).cov())
    25.0

    The support of a ``RandomProcess`` is the set of points where
    it can be evaluated.
    If no support is defined at initialization and ``randvars``
    is a sequence, it uses
    ``np.arange(0.0, len(randvars), 1.0)`` as a default.
    Before setting
    the support, it is being checked whether the length of the supports
    matches.

    >>> print(rp1.support)
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
     18. 19.]
    >>> print(rp2.support)
    [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.
     36. 38.]
    >>> rp1.support = rp2.support
    >>> print(rp1.support)
    [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.
     36. 38.]

    One can also define random processes through a map from input
    to random variable.

    >>> def rvmap(x): return RandomVariable(distribution=Normal(x, x**2))
    >>> rp_map = RandomProcess(rvmap)
    >>> print(rp_map(0.))
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp_map(1.).mean())
    1.0

    In this case, the support is empty and the bounds are the entire
    real line.

    >>> print(rp_map.support)
    None
    >>> print(rp_map.bounds)
    (-inf, inf)


    """
    def __init__(self, randvars, support=None, bounds=None):
        """
        Random process as a sequence of random variables.
        """

        # todo: refine the below. ATM it is ugly AF.
        #  Though make it work first.

        # todo: check that if rvmap is a seq, all dtypes and shapes
        #  coincide and set self._shape and self._dtype accordingly.

        support = _preprocess_support(support)
        bounds = _preprocess_bounds(bounds)
        _check_consistency_bounds_support(bounds, support)

        if callable(randvars):
            self._support = support    # None or actual values
            if bounds is not None:
                if self._support is not None:
                    if np.any(self._support < bounds[0]):
                        raise ValueError("Support must be within bounds")
                    elif np.any(self._support > bounds[1]):
                        raise ValueError("Support must be within bounds")
                bounds = bounds
            else:
                if self._support is not None and self._support.ndim > 1:
                    errormsg = "Please specify bounds for " \
                               "multidimensional random processes."
                    raise ValueError(errormsg)
                bounds = (-np.inf, np.inf)
        else:  # randvars is seq
            if support is None:
                self._support = np.arange(start=0, stop=len(randvars), step=1.0)
            else:
                if len(support) != len(randvars):
                    errormsg = ("Size of support must match "
                                "size of rvcoll")
                    raise ValueError(errormsg)
                self._support = support
            bounds = (min(self._support), max(self._support))

        super().__init__(bounds=bounds, dtype=np.nan)
        self._randvars = randvars

    # Callable type methods ############################################

    def __call__(self, x):
        """
        Evaluate random process at :math:`x`.
        """
        if callable(self._randvars):
            return self._randvars(x)
        else:
            try:
                indices = np.where(self._support == x)[0][0]
                # print(self._rvcoll[indices])
                # return 0
                return self._randvars[indices]
            except ValueError:
                errormsg = ("Random process is not supported "
                            "at that point")
                raise ValueError(errormsg)

    # Container type methods ###########################################

    def __len__(self):
        """
        The length of the process is the length of the array of
        random variables.
        """
        if self._support is None:
            raise NotImplementedError
        else:
            return len(self._support)

    def __getitem__(self, index):
        """
        Get the i-th item which is a random variable.
        """
        if callable(self._randvars):
            raise NotImplementedError
        else:
            return self._randvars[index]

    def __setitem__(self, index, randvar):
        """
        Set the i-th item which is a random variable.
        """
        if callable(self._randvars):
            raise NotImplementedError
        else:
            self._randvars[index] = randvar

    def __contains__(self, item):
        """
        """
        if callable(self._randvars):
            raise NotImplementedError
        else:
            return item in self._randvars

    def sort(self):
        """
        Sort the support points and associated random variables.
        """
        if callable(self._randvars):
            raise NotImplementedError
        else:
            raise NotImplementedError("Todo")

    # Numeric type methods (binary) ####################################

    def __add__(self, other):
        return NotImplemented

    def __sub__(self, other):
        return NotImplemented

    def __mul__(self, other):
        return NotImplemented

    def __truediv__(self, other):
        return NotImplemented

    def __pow__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return NotImplemented

    def __rmul__(self, other):
        return NotImplemented

    def __rtruediv__(self, other):
        return NotImplemented

    def __rpow__(self, other):
        return NotImplemented

    # Numeric type methods (unary) ####################################

    def __neg__(self, other):
        return NotImplemented

    def __pos__(self, other):
        return NotImplemented

    # Properties and setters  ##########################################

    @property
    def support(self):
        """
        Support of the random process.
        """
        return self._support

    @support.setter
    def support(self, support):
        """
        Support of the random process.
        """
        if callable(self._randvars):
            raise NotImplementedError("Random process is continuous.")
        if len(self._support) != len(support):
            errormsg = "Size of support does not fit RandomProcess."
            raise ValueError(errormsg)
        self._support = support

    # Statistics functions #############################################

    def meanfun(self, x):
        """
        Evaluate random process at :math:`x` and return the mean
        of the resulting distribution.
        """
        rv = self.__call__(x)
        try:
            return rv.mean()
        except NotImplementedError:
            errormsg = ("Mean of random process "
                        "is not implemented at x")
            raise NotImplementedError(errormsg)

    def covfun(self, x):
        """
        Evaluate random process at :math:`x` and return the covariance
        of the resulting distribution.
        """
        rv = self.__call__(x)
        try:
            return rv.cov()
        except NotImplementedError:
            errormsg = ("Covariance of random process "
                        "is not implemented at x")
            raise NotImplementedError(errormsg)

    def sample(self, x, size=()):
        """
        Sample from the random process at location :math:`x`.
        """
        return self.__call__(x).sample(size=size)


def _check_consistency_bounds_support(bounds, support):
    """
    """
    if bounds is not None:
        if bounds.ndim == 1:
            if support is not None and support.ndim > 1:
                errormsg = ("Please provide support and bounds "
                            "of the same dimensionality")
                raise ValueError(errormsg)
        else:
            if support.shape[1] != bounds.shape[0]:
                errormsg = ("Please provide support and bounds "
                            "of the same dimensionality")
                raise ValueError(errormsg)


def _preprocess_bounds(bounds):
    """
    """
    if bounds is not None:
        bounds = np.array(bounds)
        if bounds.ndim == 1:  # 1d inputs
            if len(bounds) != 2:
                errormsg = ("Please provide bounds with "
                            "shape (d, 2) or (2,)")
                raise ValueError(errormsg)
            if bounds[1] < bounds[0]:
                errormsg = ("Please provide bounds with "
                            "bounds[0] < bounds[1]")
                raise ValueError(errormsg)
        else:  # nd inputs
            if bounds.shape[1] != 2:
                errormsg = ("Please provide bounds with "
                            "shape (d, 2) or (2,)")
                raise ValueError(errormsg)
    return bounds


def _preprocess_support(support):
    """
    """
    if support is not None:
        support = np.array(support)
        if not np.issubdtype(support.dtype, np.number):
            raise ValueError("dtype of support must be a number")
        if support.ndim == 0:
            support = support.reshape((1,))
    return support


























def asrandproc(obj):
    """
    Wraps obj as a RandomProcess.
    """
    # todo: wrap asrandvar() into asrandproc for sequences
    #  and figure out how to do it well for callables.
    raise NotImplementedError("todo")


if __name__ == "__main__":

    # todo: turn these bad boys below into unittests

    from probnum.prob import RandomVariable, Normal

    rvs = [RandomVariable(distribution=Normal()) for i in range(10)]
    print()

    # One-dimensional input space ######################################
    # todo: do same tests in higher dimensional input

    # Sequence of RVs, no support, no bounds
    rp = RandomProcess(rvs)
    print(rp.support)  # [0, ..., 9]
    print(rp.bounds)  # (0, 9)
    print()

    # Sequence of RVs, support, no bounds
    supp = [-1.23 + 0.1*i for i in range(10)]
    rp = RandomProcess(rvs, support=supp)
    print(rp.support)  # [-1.23, -1.22, ..., -0.33]
    print(rp.bounds)  # (-1.23, -0.33)
    print()

    # Sequence of RVs, no support, bounds
    bds = (-3, 100)
    rp = RandomProcess(rvs, bounds=bds)
    print(rp.support)  # [-3, ..., 89.7]
    print(rp.bounds)  # (3, 100)
    print()

    # Sequence of RVs, support, bounds (work together)
    supp = [-1.23 + 0.1*i for i in range(10)]
    rp = RandomProcess(rvs, support=supp, bounds=bds)
    print(rp.support)  # [-1.23, -1.22, ..., -0.33]
    print(rp.bounds)  # (3, 100)
    print()

    # Sequence of RVs, support, bounds (not work together)
    supp = [-1.23 + 0.1*i for i in range(10)]
    bds = (3, 100)
    try:
        rp = RandomProcess(rvs, support=supp, bounds=bds)
    except AssertionError:
        print("Exception for mismatch worked.")
    print()

    def rvmap(x): return RandomVariable(distribution=Normal(x, 0.1))

    # callable of rvs, no support, no bds
    rp = RandomProcess(rvmap)
    print(rp.support)  # None
    print(rp.bounds)  # (-inf, inf)
    print()

    # callable of rvs, support, no bds
    supp = [-1.23 + 0.1*i for i in range(10)]
    rp = RandomProcess(rvmap, support=supp)
    print(rp.support)  # [-1.23, -1.22, ..., -0.33]
    print(rp.bounds)  # (-inf, inf)
    print()

    # callable of rvs, no support, bds
    bds = (-3, 100)
    rp = RandomProcess(rvmap, bounds=bds)
    print(rp.support)  # None
    print(rp.bounds)  # (-3, 100)
    print()

    # callable of rvs, support, bds (work together)
    bds = (-3, 100)
    supp = [-1.23 + 0.1*i for i in range(10)]
    rp = RandomProcess(rvs, support=supp, bounds=bds)
    print(rp.support)  # [-1.23, -1.22, ..., -0.33]
    print(rp.bounds)  # (-3, 100)
    print()

    # callable of rvs, support, bds (work together)
    supp = [-1.23 + 0.1*i for i in range(10)]
    bds = (3, 100)
    try:
        rp = RandomProcess(rvs, support=supp, bounds=bds)
    except AssertionError:
        print("Exception for mismatch worked.")
    print()


