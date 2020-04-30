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
    Random processes.

    Random processes are the in- and output of probabilistic
    numerical algorithms involving time- and space-components.

    Per definition, random processes are a (potentially uncountable)
    collection of random variables. More precisely, a random process
    (random field) is a map from an element in some topological space
    to a random variable.
    Therefore, the RandomProcess interface is like a function.
    Further, if the collection is finite, RandomProcess
    behaves like a sequence: it is sliceable, loopable and so on.

    Rule of thumb for parameter choices:
    In general, the object assumes to be one-dimensional unless either
    support or bounds say otherwise. It further assumes to be discrete
    as soon as randvars is a sequence (bounds given by support)
    and assumes to be continuous as soon as randvars is a callable
    (bounds are -inf to inf). If any parameter is initialised, these
    assumptions are influenced accordingly.

    The RandomProcess data structure simultaneously emulates
    basic callable objects, basic container types and basic numerict
    types (essentially: arrays).

    Parameters
    ----------
    randvars : seq or callable
        Collection of variables. Either defined as a sequence
        [rv1, rv2, ..., rvN] or as a map x -> rv(x).
    support : seq, optional.
        Support points of the random process.
        Expects shape (len(rvmap), ndim) if rvmap is a sequence.
        For instance, in a sequence of 9 randvars in d=2 needs 9 support
        points [(a1, b1), (a1, b2), (a1, b3), ..., (a3, b2), (a3, b3)]
        If ``support`` is specified, the object automatically
        assumes that the process is discrete.
        If ``support`` is not specified, there are two potential
        outcomes.
        1. rvmap is a sequence:
        ``support`` is set to be [0, 1, ..., len(randvar)-1],
        i.e. list(range(len(rvmap))). This automatically assumes that
        the random process only depends on a single variable.
        2. rvmap is a callable:
        the object assumes that the process is continuous in which
        case the support key is left empty.
    bounds : seq, optional.
        Bounds of the random process.
        Expects shape (ndim, 2), respectively (2,) if ndim is empty.
        Lower and upper bounds of the support for each dimension.
        If left empty there are three potential outcomes.
        1. rvmap is a sequence and support is specified:
        bounds are taken to be min and max of the support points
        in each respective dimension
        2. rvmap is a sequence and no support is specified:
        bounds are taken to be (0, len(rvmap)), assuming that the
        process only depends on a single variable.
        3. rvmap is a callable and no support is specified:
        Bounds are set to (-inf, inf), assuming that the process
        is continuous on the entire real line.

    Raises
    ------
    ValueError
        If dimensions of bounds and support do not match.

    Examples
    --------
    Initialise a RandomProcess with a finite collection of
    random variables. Here, we use a list of RandomVariable objects
    with Normal distribution.

    >>> import numpy as np
    >>> from probnum.prob.randomprocess import RandomProcess
    >>> from probnum.prob import RandomVariable, Normal
    >>> rvs = [RandomVariable(distribution=Normal(idx, idx**2))
    ...        for idx in range(20)]
    >>> rp1 = RandomProcess(rvs)
    >>> supp = list(range(0, 40, 2))
    >>> rp2 = RandomProcess(rvs, support=supp)

    In this case, RandomProcess behaves like a sequence itself.

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

    If called directly at input `x`, they return the random variable
    representing the process at time `x`.

    >>> print(rp1(2))
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp2(10))
    <() RandomVariable with dtype=<class 'float'>>

    Its support is the number of points where it can be evaluated.
    If no support is defined at initialization, it uses
    `list(range(len(randvar))` as a default value. Before setting
    the support, it is being checked whether the length of the supports
    matches.

    >>> print(rp1.support)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    >>> print(rp2.support)
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    >>> rp1.support = rp2.support
    >>> print(rp1.support)
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    >>> rp1.support = rp2.support.extend(100)
    ValueError: Size of support does not fit this RandomProcess.

    One can append and extend random variables

    >>> newrv = RandomVariable(distribution=Normal())
    >>> rp1.append(newrv, rp1.support[-1]+1)
    >>> rp1.extend(rp2)

    as well as do all kinds of other cool things.
    """
    def __init__(self, randvars, support=None, bounds=None):
        """
        Random process as a sequence of random variables.
        """

        # todo: refine the below. ATM it is ugly AF

        self._randvars = randvars
        if callable(randvars):
            if support is None and bounds is None:
                self._support = None
                self._bounds = (-np.inf, np.inf)
            elif support is None and bounds is not None:
                self._support = None
                self._bounds = bounds
            elif support is not None and bounds is None:
                self._support = support
                self._bounds = (-np.inf, np.inf)
            else:
                pass
        else:  # randvars is seq
            if support is None and bounds is None:
                self._support = list(range(len(randvars)))
                self._bounds = (min(self._support), max(self._support))
            elif support is None and bounds is not None:
                step = (bounds[1] - bounds[0]) / len(randvars)
                self._support = list(np.arange(bounds[0], bounds[1], step))
                self._bounds = bounds
            elif support is not None and bounds is None:
                assert len(support) == len(randvars)
                self._support = list(support)
                self._bounds = (min(self._support), max(self._support))
            else:
                assert np.all(np.array(support) > bounds[0])
                assert np.all(np.array(support) < bounds[1])
                assert len(support) == len(randvars)
                self._support = list(support)
                self._bounds = bounds

    # Callable type methods ############################################

    def __call__(self, x):
        """
        Find index of x==self.support and return corresponding random
        variable.
        """
        try:
            return self._randvars[self._support.index(x)]
        except ValueError:
            errormsg = "Random process is not supported at that point"
            raise ValueError(errormsg)

    # Container type methods ###########################################

    def __len__(self):
        """
        The length of the process is the length of the array of
        random variables.
        """
        return len(self._randvars)

    def __getitem__(self, index):
        """
        Get the i-th item which is a random variable.
        """
        return self._randvars[index]

    def __setitem__(self, index, randvar):
        """
        Set the i-th item which is a random variable.
        """
        self._randvars[index] = randvar

    def __contains__(self, item):
        """
        """
        return item in self._randvars

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
        """
        return self._support

    @support.setter
    def support(self, supp):
        """
        """
        if len(self._support) != len(supp):
            errormsg = "Size of support does not fit RandomProcess."
            raise ValueError(errormsg)
        self._support = supp

    @property
    def bounds(self):
        """
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bds):
        """
        """
        if len(self._bounds) != len(bds):  # incomplete check!!
            errormsg = "Size of bounds does not fit RandomProcess."
            raise ValueError(errormsg)
        self._bounds = bds

    @property
    def dtype(self):
        """
        """
        return self._randvars[0].dtype

    @dtype.setter
    def dtype(self, dtype):
        """
        """
        raise NotImplementedError("TODO")

    @property
    def shape(self):
        """
        """
        return self._randvars[0].shape

    @shape.setter
    def shape(self, shape):
        """
        """
        raise NotImplementedError("TODO")

    # Statistics functions #############################################

    def meanfun(self, x):
        rv = self.__call__(x)
        try:
            return rv.mean()
        except NotImplementedError:
            errormsg = ("Mean of random process "
                        "is not implemented at x")
            raise NotImplementedError(errormsg)

    def covfun(self, x):
        """ """
        rv = self.__call__(x)
        try:
            return rv.cov()
        except NotImplementedError:
            errormsg = ("Covariance of random process "
                        "is not implemented at x")
            raise NotImplementedError(errormsg)

    def sample(self, x, size=()):
        """ """
        return self.__call__(x).sample(size=size)


















def asrandproc(obj):
    """
    Wraps obj as a RandomProcess.

    Parameters
    ----------
    obj

    Returns
    -------

    """
    # todo: wrap asrandvar() into asrandproc for sequences
    #  and figure out how to do it well for callables.
    pass


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


