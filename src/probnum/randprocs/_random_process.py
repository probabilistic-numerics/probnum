"""Random Processes."""

import abc
from typing import Callable, Generic, Type, TypeVar, Union

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
            *shape=* ``batch_shape + `` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the random process. Currently, we require ``batch_shape``
            to have at most one dimension.

        Returns
        -------
        randvars.RandomVariable
            *shape=* ``batch_shape +`` :attr:`output_shape` -- Random process evaluated
            at the input(s).
        """
        raise NotImplementedError

    @property
    def input_shape(self) -> ShapeType:
        """Shape of inputs to the random process."""
        return self._input_shape

    @property
    def output_shape(self) -> ShapeType:
        """Shape of the random process evaluated at an input."""
        return self._output_shape

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) the random process evaluated at an input."""
        return self._dtype

    def marginal(self, args: _InputType) -> randvars._RandomVariableList:
        """Batch of random variables defining the marginal distributions at the inputs.

        Parameters
        ----------
        args
            *shape=* ``batch_shape + `` :attr:`input_shape` -- (Batch of) input(s) at
            which to evaluate the random process. Currently, we require ``batch_shape``
            to have at most one dimension.
        """
        # return self.__call__(args).marginal()
        raise NotImplementedError

    @property
    def mean(self) -> _function.Function:
        r"""Mean function :math:`m(x) = \mathbb{E}[f(x)]` of the random process"""
        raise NotImplementedError

    @property
    def cov(self) -> kernels.Kernel:
        r"""Covariance function :math:`k(x_0, x_1) = \mathbb{E}[(f(x_0) - \mathbb{E}[
        f(x_0)])(f(x_0) - \mathbb{E}[f(x_0)])^\top]` of the random process."""
        raise NotImplementedError

    def var(self, args: _InputType) -> _OutputType:
        """Variance function.

        Returns the variance function which is the value of the covariance or kernel
        evaluated elementwise at ``args`` for each output dimension separately.

        Parameters
        ----------
        args
            *shape=* ``batch_shape + input_shape_bcastable`` -- (Batch of) input(s) at
            which to evaluate the variance function. ``input_shape_bcastable`` must be a
            shape that can be broadcast to :attr:`input_shape`.

        Returns
        -------
        _OutputType
            *shape=* ``batch_shape`` or ``output_shape[:1] + batch_shape`` -- Variance
            of the process at ``args``.
        """
        try:
            var = self.cov(args, None)
        except NotImplementedError as exc:
            raise NotImplementedError from exc

        assert (
            var.shape
            == 2 * self._output_shape + args.shape[: args.ndim - self._input_ndim]
        )

        if self._output_ndim == 0:
            return var

        assert self._output_ndim == 1

        return np.diagonal(var, axis1=0, axis2=1)

    def std(self, args: _InputType) -> _OutputType:
        """Standard deviation function.

        Parameters
        ----------
        args
            *shape=* ``batch_shape + input_shape_bcastable`` -- (Batch of) input(s) at
            which to evaluate the standard deviation function. ``input_shape_bcastable``
            must be a shape that can be broadcast to :attr:`input_shape`.

        Returns
        -------
        _OutputType
            *shape=* ``batch_shape`` or ``output_shape[:1] + batch_shape`` -- Standard
            deviation of the process at ``args``.
        """
        try:
            return np.sqrt(self.var(args=args))
        except NotImplementedError as exc:
            raise NotImplementedError from exc

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
            *shape=* ``sample_shape + `` :attr:`input_shape` -- (Batch of) input(s) at
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
            *shape=* ``size + `` :attr:`input_shape` -- (Batch of) input(s) at
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
            *shape=* ``size + `` :attr:`input_shape` -- (Batch of) input(s) at
            which the sample paths will be evaluated. Currently, we require
            ``size`` to have at most one dimension.
        size
            Size of the sample.
        """
        raise NotImplementedError
