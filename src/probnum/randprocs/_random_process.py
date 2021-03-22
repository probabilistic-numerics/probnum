"""Random Processes."""

import abc
from typing import Callable, Generic, Optional, TypeVar, Union

import numpy as np

from probnum import randvars
from probnum import utils as _utils
from probnum.type import DTypeArgType, IntArgType, RandomStateArgType, ShapeArgType

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
    GaussMarkovProcess : Gaussian processes with the Markov property.

    Notes
    -----
    Random processes are assumed to have an (un-/countably) infinite domain. Random
    processes with a finite index set are represented by :class:`RandomVariable` s.
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
    def __call__(self, x: _InputType) -> randvars.RandomVariable[_OutputType]:
        """Evaluate the random process at a set of inputs.

        Parameters
        ----------
        x
            *shape=(input_dim,) or (n, input_dim)* -- Input(s) to evaluate random
            process at.

        Returns
        -------
        f
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

    @abc.abstractmethod
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
            *shape=(), (output_dim, ) or (n, output_dim)* -- Mean function of the
            process evaluated at inputs ``x``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cov(self, x0: _InputType, x1: Optional[_InputType] = None) -> _OutputType:
        r"""Covariance function or kernel.

        Returns the covariance function :math:`\operatorname{Cov}(f(x_0),
        f(x_1)) = \mathbb{E}[(f(x_0) - \mathbb{E}[f(x_0)])(f(x_0) - \mathbb{E}[f(
        x_0)])^\top]` of the process evaluated at ``x0`` and ``x1``. If only ``x0`` is
        given the covariance among the components of the random process at the
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
            *shape=(), (output_dim, output_dim), (n0, n1) or (n0, n1, output_dim,
            output_dim)* -- Covariance of the process at ``x0`` and ``x1``. If
            only ``x0`` is given the kernel matrix :math:`K=k(X_0, X_0)` is computed.
        """  # pylint: disable=trailing-whitespace
        raise NotImplementedError

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
            *shape=(), (output_dim,) or (n, output_dim)* -- Variance of the
            process at ``x``.
        """
        try:
            cov = self.cov(x0=x)
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
            *shape=(), (output_dim,) or (n, output_dim)* -- Standard deviation of the
            process at ``x``.
        """
        try:
            return np.sqrt(self.var(x=x))
        except NotImplementedError as exc:
            raise NotImplementedError from exc

    def sample(
        self,
        x: _InputType = None,
        size: ShapeArgType = (),
        random_state: RandomStateArgType = None,
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
        random_state :
            Random state of the random process. If None (or np.random), the global
            :mod:`numpy.random` state is used. If integer, it is used to seed the local
            :class:`~numpy.random.RandomState` instance.
        """
        if x is None:
            return lambda x0: self._sample_at_input(
                x=x0, size=size, random_state=random_state
            )

        return self._sample_at_input(x=x, size=size, random_state=random_state)

    @abc.abstractmethod
    def _sample_at_input(
        self,
        x: _InputType,
        size: ShapeArgType = (),
        random_state: RandomStateArgType = None,
    ) -> _OutputType:
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
        random_state :
            Random state of the random process. If None (or np.random), the global
            :mod:`numpy.random` state is used. If integer, it is used to seed the local
            :class:`~numpy.random.RandomState` instance.
        """
        raise NotImplementedError
