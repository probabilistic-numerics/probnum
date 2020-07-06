"""
Random / Stochastic processes.

This module implements random processes as (potentially uncountable)
collections of random variables.

Incomplete
----------
* Implementing some random_state's (could use some help here)
* Unittests (matter of time)
* Documentation (matter of time)
"""

from abc import ABC, abstractmethod
import numpy as np


class _RandomProcess(ABC):
    """
    Interface for random processes.
    The object `RandomProcess` is merely a factory.

    Certain methods are @abstractmethods for
    1) robustness: Refactoring subclass does not break interface.
    2) added documentation: the things that distinguish continuous-
       and discrete-time implementations are precisely the abstract
       methods in _RandomProcess.
    """
    def __init__(self, randvars, supportpts, bounds):
        """ """
        self._type_check(bounds, randvars, supportpts)
        self._randvars = randvars
        self._supportpts = supportpts
        self._bounds = bounds

    def _type_check(self, bounds, randvars, supportpts):
        """
        Always true if __init__ is called through either
        _DiscreteProcess or _ContinuousProcess.
        """
        assert isinstance(randvars, np.ndarray) or callable(randvars)
        assert isinstance(supportpts, np.ndarray) or supportpts is None
        assert isinstance(bounds, np.ndarray) or bounds is None

    # Abstract callable type methods ###################################

    @abstractmethod
    def __call__(self, x):
        """ """
        raise NotImplementedError

    # Abstract statistical functions ###################################

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

    # Available properties, getters and setters ########################

    @property
    def randvars(self):
        """
        """
        return self._randvars

    # Abstract properties, getters and setters #########################

    @property
    @abstractmethod
    def domain(self):
        """
        Domain of the random process. Either discrete set of points
        or a range of intervals.
        """
        raise NotImplementedError

    @domain.setter
    @abstractmethod
    def domain(self, domain):
        """
        Domain of the random process. Either discrete set of points
        or a range of intervals.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def supportpts(self):
        """
        Support points of a discrete random process.
        """
        return self._supportpts

    @supportpts.setter
    @abstractmethod
    def supportpts(self, supportpts):
        """
        Support points of a discrete random process.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def bounds(self):
        """
        Bounds of the support of a continuous random process.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """
        Bounds of the support of a continuous random process.
        """
        raise NotImplementedError

    # Container type methods ###########################################
    # Implemented for discrete processes only ##########################

    def __len__(self):
        """
        Length of the array of random variables.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Get the i-th item which is a random variable.
        """
        raise NotImplementedError

    def __setitem__(self, index, randvar):
        """
        Set the i-th item which is a random variable.
        """
        raise NotImplementedError

    def __contains__(self, item):
        """
        """
        raise NotImplementedError

    # Numeric type methods (binary) ####################################
    # Might be implemented at some point in the future #################

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


class RandomProcess(_RandomProcess):
    """
    Random processes are the in- and output of probabilistic
    numerical algorithms involving time- and space-components.

    A random process is a collection of random variables
    :math:`X = \\{X_{\\tau} \\}_{\\tau \\in \\mathcal{T}}`
    indexed by some topological space :math:`\\mathcal{T}`.
    More precisely, a
    random process (random field) can be thought of as a map

    .. math:: X : \\mathcal{T} \\longrightarrow RV,
        \\quad \\tau \\longmapsto X_{\\tau}

    from an element in a topological space to a random variable.
    For this reason the RandomProcess interface behaves like a function.

    The object assumes to be one-dimensional unless either
    ``support`` or ``bounds`` say otherwise.
    Any variable that is ignored is set to ``None``.

    - *Discrete:* As soon as ``support`` is specified, the object
      assumes to be discrete and ``bounds`` is ignored.

    - *Continuous:* If no ``support`` is specified, the object assumes
      to be continuous.
      If ``bounds`` is not specified either, ``bounds`` are set
      to ``bounds=(-inf, inf)``.


    A continuous time RandomProcess behaves like a callable and a
    numeric type. A discrete time RandomProcess additionally emulates
    container types supporting ``__len__``, ``__getitem__``, etc..


    Parameters
    ----------
    randvars : array_like or callable
        Collection of variables. Either defined as an array
        ``[rv1, rv2, ..., rvN]`` or as a map
        :math:`\\tau \\rightarrow X_\\tau`.
    supportpts : array_like, optional.
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
    >>> rp2 = RandomProcess(rvs, supportpts=supp)

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
    This only works if support points are specified.

    >>> print(rp2(10.0))
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp2(2.0).cov())
    1.0

    The domain of a discrete ``RandomProcess`` is the set of points where
    it can be evaluated.
    If a domain is not specified, this raises an Error.

    >>> print(rp2.domain)
    [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.
     36. 38.]
    >>> rp1.domain = rp2.domain
    >>> print(rp1.domain)
    [ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.
     36. 38.]

    One can also define random processes through a map from input
    to random variable.

    >>> def rvmap(x): return RandomVariable(distribution=Normal(x, x**2))
    >>> rp_map = RandomProcess(rvmap, bounds=(-np.inf, np.inf))
    >>> print(rp_map(0.))
    <() RandomVariable with dtype=<class 'float'>>
    >>> print(rp_map(1.).mean())
    1.0

    In this case, the domain is defined by the bonuds which in this
    case are the entire real line.

    >>> print(rp_map.domain)
    (-inf, inf)
    >>> print(rp_map.bounds)
    (-inf, inf)
    """
    def __new__(cls, randvars, supportpts=None, bounds=None):
        """
        Factory method.

        Either supportpts or bounds are specified.
            supportpts -> discrete process,
            bounds -> continuous process.
        If neither, the type of randvars decides.
            seq -> discrete process
            callable -> continuous
        """
        if supportpts is not None and bounds is not None:
            errormsg = "Please specify either support points or bounds."
            raise ValueError(errormsg)

        if cls is RandomProcess:
            if supportpts is not None:
                return super().__new__(_DiscreteProcess)
            elif bounds is not None:
                return super().__new__(_ContinuousProcess)
            else:
                if callable(randvars):
                    return super().__new__(_ContinuousProcess)
                else:
                    return super().__new__(_DiscreteProcess)
        else:
            return super().__new__(cls)

    def __init__(self, randvars, supportpts, bounds):
        """ """
        super().__init__(randvars, supportpts, bounds)


class _DiscreteProcess(RandomProcess, _RandomProcess):
    """
    Implementation for discrete-time processes.

    Discrete-time processes allow container-type interfaces.

    Inherits from :class:`RandomProcess` only to allow:

    >>> from probnum.prob.randomprocess import *
    >>> from probnum.prob import asrandvar
    >>> rp = RandomProcess(randvars=[asrandvar(1.0)])
    >>> issubclass(type(rp), RandomProcess)
    True
    """
    def __init__(self, randvars, supportpts=None, bounds=None):
        """ """
        randvars, supportpts = self._pre_process(randvars, supportpts)
        self._type_check(bounds, randvars, supportpts)
        super().__init__(randvars=randvars, supportpts=supportpts,
                         bounds=None)

    def _pre_process(self, randvars, supportpts):
        """
        Turns lists into arrays wherever applicable.
        """
        if not callable(randvars):
            randvars = np.array(randvars)
        if supportpts is not None:
            supportpts = np.array(supportpts)
        return randvars, supportpts

    def _type_check(self, bounds, randvars, supportpts):
        """
        Asserts that pre-processing was successful.

        If the code is called accordingly through RandomProcess,
        these checks are always true.
        """
        assert supportpts is None or isinstance(supportpts, np.ndarray)
        assert bounds is None
        assert isinstance(randvars, np.ndarray) or callable(randvars)
        if isinstance(randvars, np.ndarray):
            assert randvars.dtype == object

    # Callable type methods ############################################

    def __call__(self, x):
        """ """
        if self._supportpts is None:
            raise NotImplementedError
        else:
            indices = np.where(x == self._supportpts)[0][0]
            return self._randvars[indices]

    # Statistical functions ############################################

    def meanfun(self, x):
        """ """
        return self.__call__(x).mean()

    def covfun(self, x):
        """ """
        return self.__call__(x).cov()

    def sample(self, x, size=()):
        """ """
        return self.__call__(x).sample(size=size)

    # Properties, getters and setters ##################################

    @property
    def domain(self):
        """
        Domain of the random process. Either discrete set of points
        or a range of intervals.
        """
        if self._supportpts is None:
            errormsg = "Domain of this random process is not specified."
            raise NotImplementedError(errormsg)
        return self._supportpts

    @domain.setter
    def domain(self, domain):
        """
        Domain of the random process. Either discrete set of points
        or a range of intervals.
        """
        self._supportpts = np.array(domain)

    @property
    def supportpts(self):
        """
        Support points of a discrete random process.
        """
        return self._supportpts

    @supportpts.setter
    def supportpts(self, supportpts):
        """
        Support points of a discrete random process.
        """
        self._supportpts = np.array(supportpts)

    @property
    def bounds(self):
        """
        Bounds of the support of a continuous random process.
        """
        return None

    @bounds.setter
    def bounds(self, bounds):
        """
        Bounds of the support of a continuous random process.
        """
        raise NotImplementedError

    # Container type methods ###########################################

    def __len__(self):
        """
        Length of the array of random variables.
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


class _ContinuousProcess(RandomProcess, _RandomProcess):
    """
    Implementation for continuous-time processes.

    Inherits from :class:`RandomProcess` only to allow

    >>> from probnum.prob.randomprocess import *
    >>> from probnum.prob import asrandvar
    >>> rp = RandomProcess(randvars=(lambda x: asrandvar(x)))
    >>> issubclass(type(rp), RandomProcess)
    True
    """
    def __init__(self, randvars, supportpts=None, bounds=None):
        """ """
        bounds = self._pre_process(bounds)
        self._type_check(bounds, randvars, supportpts)
        super().__init__(randvars=randvars, supportpts=None,
                         bounds=bounds)

    def _pre_process(self, bounds):
        """ """
        if bounds is not None:
            bounds = np.array(bounds)
        else:
            bounds = np.array([-np.inf, np.inf])
        return bounds

    def _type_check(self, bounds, randvars, supportpts):
        """
        Always true if called by RandomProcess.
        """
        assert isinstance(bounds, np.ndarray) or bounds is None
        assert supportpts is None
        assert callable(randvars)

    # Callable type methods ############################################

    def __call__(self, x):
        """ """
        return self._randvars(x)

    # Statistical functions ############################################

    def meanfun(self, x):
        """ """
        return self.__call__(x).mean()

    def covfun(self, x):
        """ """
        return self.__call__(x).cov()

    def sample(self, x, size=()):
        """ """
        return self.__call__(x).sample(size=size)

    @property
    def domain(self):
        """
        Domain of the random process. Either discrete set of points
        or a range of intervals.
        """
        return self._bounds

    @domain.setter
    def domain(self, domain):
        """
        Domain of the random process. Either discrete set of points
        or a range of intervals.
        """
        self._bounds = np.array(domain)

    @property
    def supportpts(self):
        """
        Support points of a discrete random process.
        """
        return None

    @supportpts.setter
    def supportpts(self, supportpts):
        """
        Support points of a discrete random process.
        """
        raise NotImplementedError

    @property
    def bounds(self):
        """
        Bounds of the support of a continuous random process.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        """
        Bounds of the support of a continuous random process.
        """
        self._bounds = np.array(bounds)
