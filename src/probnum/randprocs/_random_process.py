"""Random Processes."""

import abc
from typing import Callable, Generic, Optional, Type, TypeVar, Union

import numpy as np

from probnum import _randomvariablelist, randvars
from probnum import utils as _utils
from probnum.typing import DTypeArgType, IntArgType, ShapeArgType

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
    input_dim :
        Input dimension of the random process.
    output_dim :
        Output dimension of the random process.
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
        input_dim: IntArgType,
        output_dim: IntArgType,
        dtype: DTypeArgType,
    ):
        self._input_dim = np.int_(_utils.as_numpy_scalar(input_dim))
        self._output_dim = np.int_(_utils.as_numpy_scalar(output_dim))
        self._dtype = np.dtype(dtype)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} with "
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"dtype={self.dtype}>"
        )

    @abc.abstractmethod
    def __call__(self, args: _InputType) -> randvars.RandomVariable[_OutputType]:
        """Evaluate the random process at a set of input arguments.

        Parameters
        ----------
        args
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to evaluate random
            process at.

        Returns
        -------
        randvars.RandomVariable
            *shape=(), (output_dim,) or (n, output_dim)* -- Random process evaluated at
            the inputs.
        """
        raise NotImplementedError

    @property
    def input_dim(self) -> int:
        """Shape of inputs to the random process."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Shape of the random process evaluated at an input."""
        return self._output_dim

    @property
    def dtype(self) -> np.dtype:
        """Data type of (elements of) the random process evaluated at an input."""
        return self._dtype

    def marginal(self, args: _InputType) -> _randomvariablelist._RandomVariableList:
        """Batch of random variables defining the marginal distributions at the inputs.

        Parameters
        ----------
        args
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to evaluate random
            process at.
        """
        # return self.__call__(args).marginal()
        raise NotImplementedError

    @abc.abstractmethod
    def mean(self, args: _InputType) -> _OutputType:
        """Mean function.

        Returns the mean function evaluated at the given input(s).

        Parameters
        ----------
        args
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) where the mean
            function is evaluated.

        Returns
        -------
        _OutputType
            *shape=(), (output_dim, ) or (n, output_dim)* -- Mean function of the
            process evaluated at inputs ``x``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cov(self, args0: _InputType, args1: Optional[_InputType] = None) -> _OutputType:
        r"""Covariance function or kernel.

        Returns the covariance function :math:`\operatorname{Cov}(f(x_0),
        f(x_1)) = \mathbb{E}[(f(x_0) - \mathbb{E}[f(x_0)])(f(x_0) - \mathbb{E}[f(
        x_0)])^\top]` of the process evaluated at :math:`x_0` and :math:`x_1`. If only
        ``args0`` is given the covariance among the components of the random process
        at the inputs defined by ``args0`` is computed.

        Parameters
        ----------
        args0
            *shape=(input_dim,) or (n0, input_dim)* -- First input to the covariance
            function.
        args1
            *shape=(input_dim,) or (n1, input_dim)* -- Second input to the covariance
            function.

        Returns
        -------
        _OutputType
            *shape=(), (output_dim, output_dim), (n0, n1) or (n0, n1, output_dim,
            output_dim)* -- Covariance of the process at ``args0`` and ``args1``. If
            only ``args0`` is given the kernel matrix :math:`K=k(x_0, x_0)` is computed.
        """  # pylint: disable=trailing-whitespace
        raise NotImplementedError

    def var(self, args: _InputType) -> _OutputType:
        """Variance function.

        Returns the variance function which is the value of the covariance or kernel
        evaluated elementwise at ``args`` for each output dimension separately.

        Parameters
        ----------
        args
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to the variance function.

        Returns
        -------
        _OutputType
            *shape=(), (output_dim,) or (n, output_dim)* -- Variance of the
            process at ``args``.
        """
        try:
            cov = self.cov(args0=args)
            if cov.ndim < 2:
                return cov
            elif cov.ndim == 2:
                return np.diag(cov)
            else:
                return np.vstack(
                    [np.diagonal(cov[:, :, i, i]) for i in range(self.output_dim)]
                ).T
        except NotImplementedError as exc:
            raise NotImplementedError from exc

    def std(self, args: _InputType) -> _OutputType:
        """Standard deviation function.

        Parameters
        ----------
        args
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to the standard
            deviation function.

        Returns
        -------
        _OutputType
            *shape=(), (output_dim,) or (n, output_dim)* -- Standard deviation of the
            process at ``args``.
        """
        try:
            return np.sqrt(self.var(args=args))
        except NotImplementedError as exc:
            raise NotImplementedError from exc

    @abc.abstractmethod
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
            *shape=(sample_size, output_dim)* -- Sample(s) from a base measure
            evaluated at the input arguments.
        """
        raise NotImplementedError

    def sample(
        self,
        rng: np.random.Generator,
        args: _InputType = None,
        size: ShapeArgType = (),
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
            *shape=(input_dim,) or (n, input_dim)* -- Evaluation input(s) of the
            sample paths of the process. If ``None``, sample paths, i.e. callables are
            returned.
        size
            Size of the sample.
        """
        if args is None:
            raise NotImplementedError

        return self._sample_at_input(rng=rng, args=args, size=size)

    @abc.abstractmethod
    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: _InputType,
        size: ShapeArgType = (),
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
            *shape=(input_dim,) or (n, input_dim)* -- Evaluation input(s) of the
            sample paths of the process.
        size
            Size of the sample.
        """
        raise NotImplementedError
