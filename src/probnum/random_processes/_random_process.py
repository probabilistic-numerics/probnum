"""Random Processes."""

from typing import Callable, Generic, Optional, TypeVar, Union

import numpy as np

import probnum.kernels as kernels
from probnum import utils as _utils
from probnum.random_variables import RandomVariable
from probnum.type import (
    DTypeArgType,
    IntArgType,
    RandomStateArgType,
    RandomStateType,
    ShapeArgType,
    ShapeType,
)

_InputType = TypeVar("InputType")
_OutputType = TypeVar("OutputType")


class RandomProcess(Generic[_InputType, _OutputType]):
    """Random processes represent uncertainty about a function.

    Random processes generalize functions by encoding uncertainty over function
    values in their covariance function. They can be used to model (deterministic)
    functions which are not fully known or to define functions with stochastic
    output.

    Instances of :class:`RandomProcess` can be added, multiplied, etc. with scalars,
    arrays and random variables. Such (arithmetic) operations may not retain all
    previously available methods.

    Parameters
    ----------
    input_dim :
        Input dimension of the random process.
    output_dim :
        Output dimension of the random process.
    dtype :
        Data type of the random process evaluated at an input. If ``object`` will be
        converted to ``numpy.dtype``.
    random_state :
        Random state of the random process. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.
    fun :
        Callable defining the random process.
    sample_at_input :
        Sampling from the random process at a set of inputs.
    mean :
        Mean function.
    cov :
        Covariance function or kernel.
    var :
        Variance function.
    std :
        Standard deviation function.

    See Also
    --------
    asrandproc : Convert an object to a random process.
    RandomVariable : Class representing random variables.

    Notes
    -----
    Random processes are assumed to have an (un-/countably) infinite domain. Random
    processes with a finite index set are represented by :class:`RandomVariable` s.

    Sampling from random processes with fixed seed is not stable with respect to the
    order of operations. This means sampling from a random process and then
    performing an arithmetic operation will not necessarily return the same samples
    as if the order of operations is reversed. However, the random seed ensures that
    each sequence of operations will always result in the same output.
    """

    # pylint: disable=too-many-instance-attributes,invalid-name

    def __init__(
        self,
        input_dim: IntArgType,
        output_dim: IntArgType,
        dtype: DTypeArgType,
        random_state: RandomStateArgType = None,
        fun: Optional[Callable[[_InputType], RandomVariable[_OutputType]]] = None,
        sample_at_input: Optional[
            Callable[[_InputType, ShapeType], _OutputType]
        ] = None,
        mean: Optional[Callable[[_InputType], _OutputType]] = None,
        cov: Optional[
            Union[Callable[[_InputType], _OutputType], kernels.Kernel]
        ] = None,
        var: Optional[Callable[[_InputType], _OutputType]] = None,
        std: Optional[Callable[[_InputType], _OutputType]] = None,
    ):
        # pylint: disable=too-many-arguments
        """Create a new random process."""

        # Function defining the random process
        self.__fun = fun

        # Dimension and data type
        self.__input_dim = int(_utils.as_numpy_scalar(input_dim))
        self.__output_dim = int(_utils.as_numpy_scalar(output_dim))
        self.__dtype = np.dtype(dtype)

        # Random seed and sampling
        self._random_state = _utils.as_random_state(random_state)
        self.__sample_at_input = sample_at_input

        # Functions of the random process
        self.__mean = mean
        self.__var = var
        self.__std = std

        # Type normalization
        if isinstance(cov, kernels.Kernel):
            if cov.input_dim != self.input_dim or cov.output_dim != self.output_dim:
                raise ValueError(
                    f"Dimensions of kernel ({cov.input_dim}, "
                    f"{cov.output_dim}) and process ({self.input_dim}, "
                    f"{self.output_dim}) do not match."
                )
            self.__cov = cov
        elif callable(cov):
            self.__cov = kernels.Kernel(
                kernelfun=cov, input_dim=self.input_dim, output_dim=self.output_dim
            )
        else:
            self.__cov = None

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"dtype={self.dtype}>"
        )

    def __call__(self, x: _InputType) -> RandomVariable[_OutputType]:
        """Evaluate the random process at a set of inputs.

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to evaluate random
            process at.

        Returns
        -------
        f
            *shape=(output_dim,) or (n, output_dim)* -- Random process evaluated at
            the inputs.
        """
        if self.__fun is None:
            raise NotImplementedError

        return self.__fun(x)

    @property
    def input_dim(self) -> int:
        """Shape of inputs to the random process."""
        return self.__input_dim

    @property
    def output_dim(self) -> int:
        """Shape of the random process evaluated at an input."""
        return self.__output_dim

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) the random process evaluated at an input."""
        return self.__dtype

    @property
    def random_state(self) -> RandomStateType:
        """Random state of the random process.

        This attribute defines the RandomState object to use for drawing
        realizations from this random variable. If None (or np.random),
        the global np.random state is used. If integer, it is used to
        seed the local :class:`~numpy.random.RandomState` instance.
        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed: RandomStateArgType):
        """Get or set the RandomState object of the random process.

        This can be either None or an existing RandomState object. If
        None (or np.random), use the RandomState singleton used by
        np.random. If already a RandomState instance, use it. If an int,
        use a new RandomState instance seeded with seed.
        """
        self._random_state = _utils.as_random_state(seed)

    def mean(self, x: _InputType) -> _OutputType:
        """Mean function.

        Returns the mean function evaluated at the given input(s).

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) where the mean
            function is evaluated.

        Returns
        -------
        mean
            *shape=(output_dim, ) or (n, output_dim)* -- Mean function of the process
            evaluated at inputs ``x``.
        """
        if self.__mean is None:
            raise NotImplementedError

        return self.__mean(x)

    def cov(self, x0: _InputType, x1: Optional[_InputType] = None) -> _OutputType:
        """Covariance function or kernel.

        Returns the covariance function :math:`\\operatorname{Cov}(f(x_0),
        f(x_1)) = \\mathbb{E}[(f(x_0) - \\mathbb{E}[f(x_0)])(f(x_0) - \\mathbb{E}[f(
        x_0)])^\\top]` of the process evaluated at ``x0`` and ``x1``. If only ``x0`` is
        provided the covariance among the components of the random process at the
        inputs defined by ``x0`` is computed.

        Parameters
        ----------
        x0
            *shape=(input_dim,) or (n0, input_dim)* -- First input to the covariance
            function.
        x1
            *shape=(input_dim,) or (n1, input_dim)* -- Second input to the covariance
            function.

        Returns
        -------
        cov
            *shape=(output_dim, output_dim) or (n0, n1) or (n0, n1, output_dim,
            output_dim)* -- Covariance of the process at ``x0`` and ``x1``. If
            only ``x0`` is given the kernel matrix :math:`K=k(X_0, X_0)` is computed.
        """  # pylint: disable=trailing-whitespace
        if self.__cov is None:
            raise NotImplementedError

        return self.__cov(x0, x1)

    def var(self, x: _InputType) -> _OutputType:
        """Variance function.

        Returns the variance function which is the value of the covariance or kernel
        evaluated elementwise at ``x`` for each output dimension separately.

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to the variance function.

        Returns
        -------
        var
            *shape=(output_dim,) or (n, output_dim)* -- Variance of the
            process at ``x``.
        """
        if self.__var is None:
            try:
                if np.atleast_2d(x).shape[0] > 1:
                    varshape = (x.shape[0], self.output_dim)
                else:
                    varshape = (self.output_dim,)
                return np.diag(self.cov(x0=x)).reshape(varshape).copy()
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            return self.__var(x)

    def std(self, x: _InputType) -> _OutputType:
        """Standard deviation function.

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to the standard
            deviation function.

        Returns
        -------
        var
            *shape=(output_dim,) or (n, output_dim)* -- Standard deviation of the
            process at ``x``.
        """
        if self.__std is None:
            try:
                return np.sqrt(self.var(x=x))
            except NotImplementedError as exc:
                raise NotImplementedError from exc
        else:
            return self.__std(x)

    def sample(
        self, x: _InputType = None, size: ShapeArgType = ()
    ) -> Union[Callable[[_InputType], _OutputType], _OutputType]:
        """Sample paths from the random process.

        If no inputs are provided this function returns sample paths which are
        callables, otherwise random variables corresponding to the input locations
        are returned.

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Evaluation input(s) of the
            sample paths of the process. If ``None``, sample paths, i.e. callables are
            returned.
        size
            Size of the sample.
        """
        if x is None:
            return lambda x0: self._sample_at_input(x=x0, size=size)

        return self._sample_at_input(x=x, size=size)

    def _sample_at_input(self, x: _InputType, size: ShapeArgType = ()) -> _OutputType:
        """Evaluate a set of sample paths at the given inputs.

        This function should be implemented by subclasses of :class:`RandomProcess`.
        This enables :meth:`sample` to both return functions, i.e. sample paths if
        only a `size` is provided and random variables if inputs are provided as well.

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Evaluation input(s) of the
            sample paths of the process.
        size
            Size of the sample.
        """
        if self.__sample_at_input is None:
            raise NotImplementedError("No sampling method provided.")

        return self.__sample_at_input(x, _utils.as_shape(size))
