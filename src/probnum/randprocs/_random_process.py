"""Random Processes."""

import abc
from typing import Callable, Generic, Optional, Type, TypeVar, Union

import numpy as np

from probnum import _function, randvars, utils as _utils
from probnum.randprocs import kernels
from probnum.typing import DTypeLike, ShapeLike, ShapeType

_InputType = TypeVar("InputType")
_OutputType = TypeVar("OutputType")


class RandomProcess(Generic[_InputType, _OutputType], abc.ABC):
    """Random processes represent uncertainty about a function.

    Random processes generalize functions by encoding uncertainty over function
    values in their covariance function. They can be used to model (deterministic)
    functions which are not fully known or to define functions with stochastic
    output.

    Parameters
    ----------
    input_shape :
        Input shape of the random process.
    output_shape :
        Output shape of the random process.
    dtype :
        Data type of the random process evaluated at an input. If ``object`` will be
        converted to ``numpy.dtype``.
    mean :
        Mean function of the random process.
    cov :
        Covariance function of the random process.

    See Also
    --------
    RandomVariable : Random variables.
    GaussianProcess : Gaussian processes.
    MarkovProcess : Random processes with the Markov property.

    Notes
    -----
    Random processes are assumed to have an (un-/countably) infinite domain. Random
    processes with a finite index set are represented by :class:`RandomVariable`.
    """

    # pylint: disable=invalid-name

    def __init__(
        self,
        input_shape: ShapeLike,
        output_shape: ShapeLike,
        dtype: DTypeLike,
        mean: Optional[_function.Function] = None,
        cov: Optional[kernels.Kernel] = None,
    ):
        self._input_shape = _utils.as_shape(input_shape)
        self._input_ndim = len(self._input_shape)

        self._output_shape = _utils.as_shape(output_shape)
        self._output_ndim = len(self._output_shape)

        if self._output_ndim > 1:
            raise ValueError(
                "Currently, we only support random processes with at most one output"
                "dimension."
            )

        self._dtype = np.dtype(dtype)

        # Mean function
        if mean is not None:
            if not isinstance(mean, _function.Function):
                raise TypeError("The mean function must have type `probnum.Function`.")

            if mean.input_shape != self._input_shape:
                raise ValueError(
                    f"The mean function must have the same `input_shape` as the random "
                    f"process (`{mean.input_shape}` != `{self._input_shape}`)."
                )

            if mean.output_shape != self._output_shape:
                raise ValueError(
                    f"The mean function must have the same `output_shape` as the "
                    f"random process (`{mean.output_shape}` != `{self._output_shape}`)."
                )

        self._mean = mean

        # Covariance function
        if cov is not None:
            if not isinstance(cov, kernels.Kernel):
                raise TypeError(
                    "The covariance functions must be implemented as a " "`Kernel`."
                )

            if cov.input_shape != self._input_shape:
                raise ValueError(
                    f"The covariance function must have the same `input_shape` as the "
                    f"random process (`{cov.input_shape}` != `{self._input_shape}`)."
                )

            if cov.output_shape != 2 * self._output_shape:
                raise ValueError(
                    f"The `output_shape` of the covariance function must be given by "
                    f"`2 * self.output_shape` (`{cov.output_shape}` != "
                    f"`{2 * self._output_shape}`)."
                )

        self._cov = cov

    @property
    def input_shape(self) -> ShapeType:
        """Shape of inputs to the random process."""
        return self._input_shape

    @property
    def input_ndim(self) -> int:
        """Syntactic sugar for ``len(input_shape)``."""
        return self._input_ndim

    @property
    def output_shape(self) -> ShapeType:
        """Shape of the random process evaluated at an input."""
        return self._output_shape

    @property
    def output_ndim(self) -> int:
        """Syntactic sugar for ``len(output_shape)``."""
        return self._output_ndim

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) the random process evaluated at an input."""
        return self._dtype

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"input_shape={self.input_shape}, output_shape={self.output_shape}, "
            f"dtype={self.dtype}>"
        )

    @abc.abstractmethod
    def __call__(self, args: _InputType) -> randvars.RandomVariable[_OutputType]:
        """Evaluate the random process at a set of input arguments.

        Parameters
        ----------
        args
            *shape=* ``batch_shape +`` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the random process. Currently, we require ``batch_shape``
            to have at most one dimension.

        Returns
        -------
        randvars.RandomVariable
            *shape=* ``batch_shape +`` :attr:`output_shape` -- Random process evaluated
            at the input(s).
        """

    def marginal(self, args: _InputType) -> randvars._RandomVariableList:
        """Batch of random variables defining the marginal distributions at the inputs.

        Parameters
        ----------
        args
            *shape=* ``batch_shape +`` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the random process. Currently, we require ``batch_shape``
            to have at most one dimension.
        """
        raise NotImplementedError

    @property
    def mean(self) -> _function.Function:
        r"""Mean function :math:`m(x) := \mathbb{E}[f(x)]` of the random process."""
        if self._mean is None:
            raise NotImplementedError

        return self._mean

    @property
    def cov(self) -> kernels.Kernel:
        r"""Covariance function :math:`k(x_0, x_1)` of the random process.

        .. math::
            :nowrap:

            \begin{equation}
                k(x_0, x_1) := \mathbb{E} \left[
                    (f(x_0) - \mathbb{E}[f(x_0)])
                    (f(x_1) - \mathbb{E}[f(x_1)])^\top
                \right]
            \end{equation}
        """
        if self._cov is None:
            raise NotImplementedError

        return self._cov

    def var(self, args: _InputType) -> _OutputType:
        """Variance function.

        Returns the variance function which is the value of the covariance or kernel
        evaluated elementwise at ``args`` for each output dimension separately.

        Parameters
        ----------
        args
            *shape=* ``batch_shape +`` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the variance function.

        Returns
        -------
        _OutputType
            *shape=* ``batch_shape +`` :attr:`output_shape` -- Variance of the process
            at ``args``.
        """
        pointwise_covs = self.cov(args, None)  # pylint: disable=not-callable

        assert (
            pointwise_covs.shape
            == args.shape[: args.ndim - self._input_ndim] + 2 * self._output_shape
        )

        if self._output_ndim == 0:
            return pointwise_covs

        assert self._output_ndim == 1

        return np.diagonal(pointwise_covs, axis1=-2, axis2=-1)

    def std(self, args: _InputType) -> _OutputType:
        """Standard deviation function.

        Parameters
        ----------
        args
            *shape=* ``batch_shape +`` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the standard deviation function.

        Returns
        -------
        _OutputType
            *shape=* ``batch_shape +`` :attr:`output_shape` -- Standard deviation of the
            process at ``args``.
        """
        return np.sqrt(self.var(args=args))

    def push_forward(
        self,
        args: _InputType,
        base_measure: Type[randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        """Transform samples from a base measure into samples from the random process.

        This function can be used to control sampling from the random process by
        explicitly passing samples from a base measure evaluated at the input arguments.

        Parameters
        ----------
        args
            Input arguments.
        base_measure
            Base measure. Given as a type of random variable.
        sample
            *shape=* ``sample_shape +`` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the random process. Currently, we require ``sample_shape``
            to have at most one dimension.
        """
        raise NotImplementedError

    def sample(
        self,
        rng: np.random.Generator,
        args: _InputType = None,
        size: ShapeLike = (),
    ) -> Union[Callable[[_InputType], _OutputType], _OutputType]:
        """Sample paths from the random process.

        If no inputs are provided this function returns sample paths which are
        callables, otherwise random variables corresponding to the input locations
        are returned.

        Parameters
        ----------
        rng
            Random number generator.
        args
            *shape=* ``size +`` :attr:`input_shape` -- (Batch of) input(s) at
            which the sample paths will be evaluated. Currently, we require
            ``size`` to have at most one dimension. If ``None``, sample paths,
            i.e. callables are returned.
        size
            Size of the sample.
        """
        if args is None:
            raise NotImplementedError

        return self._sample_at_input(rng=rng, args=args, size=size)

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: _InputType,
        size: ShapeLike = (),
    ) -> _OutputType:
        """Evaluate a set of sample paths at the given inputs.

        This function should be implemented by subclasses of :class:`RandomProcess`.
        This enables :meth:`sample` to both return functions, i.e. sample paths if
        only a `size` is provided and random variables if inputs are provided as well.

        Parameters
        ----------
        rng
            Random number generator.
        args
            *shape=* ``size +`` :attr:`input_shape` -- (Batch of) input(s) at
            which the sample paths will be evaluated. Currently, we require
            ``size`` to have at most one dimension.
        size
            Size of the sample.
        """
        return self(args).sample(rng, size=size)
