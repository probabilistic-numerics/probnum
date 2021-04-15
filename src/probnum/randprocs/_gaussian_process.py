"""Gaussian processes."""

from typing import Callable, Optional, Type, Union

import numpy as np

from probnum import kernels, randvars
from probnum.type import RandomStateArgType, ShapeArgType

from . import _random_process

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussianProcess(_random_process.RandomProcess[_InputType, _OutputType]):
    """Gaussian processes.

    A Gaussian process is a continuous stochastic process which if evaluated at a
    finite set of inputs returns a random variable with a normal distribution. Gaussian
    processes are fully characterized by their mean and covariance function.

    Parameters
    ----------
    mean :
        Mean function.
    cov :
        Covariance function or kernel.

    See Also
    --------
    RandomProcess : Random processes.
    MarkovProcess : Random processes with the Markov property.

    Examples
    --------
    Define a Gaussian process with a zero mean function and RBF kernel.

    >>> import numpy as np
    >>> from probnum.kernels import ExpQuad
    >>> from probnum.randprocs import GaussianProcess
    >>> mu = lambda x : np.zeros_like(x)  # zero-mean function
    >>> k = ExpQuad(input_dim=1)  # RBF kernel
    >>> gp = GaussianProcess(mu, k)

    Sample from the Gaussian process.

    >>> x = np.linspace(-1, 1, 5)[:, None]
    >>> np.random.seed(42)
    >>> gp.sample(x)
    array([[-0.35187364],
           [-0.41301096],
           [-0.65094306],
           [-0.56817194],
           [ 0.01173088]])
    >>> gp.cov(x)
    array([[1.        , 0.8824969 , 0.60653066, 0.32465247, 0.13533528],
           [0.8824969 , 1.        , 0.8824969 , 0.60653066, 0.32465247],
           [0.60653066, 0.8824969 , 1.        , 0.8824969 , 0.60653066],
           [0.32465247, 0.60653066, 0.8824969 , 1.        , 0.8824969 ],
           [0.13533528, 0.32465247, 0.60653066, 0.8824969 , 1.        ]])
    """

    def __init__(
        self,
        mean: Callable[[_InputType], _OutputType],
        cov: kernels.Kernel,
    ):
        if not isinstance(cov, kernels.Kernel):
            raise TypeError(
                "The covariance functions must be implemented as a " "`Kernel`."
            )

        self._meanfun = mean
        self._covfun = cov
        super().__init__(
            input_dim=cov.input_dim,
            output_dim=cov.output_dim,
            dtype=np.dtype(np.float_),
        )

    def __call__(self, args: _InputType) -> randvars.Normal:
        return randvars.Normal(mean=self.mean(args), cov=self.cov(args))

    def mean(self, args: _InputType) -> _OutputType:
        return self._meanfun(args)

    def cov(self, args0: _InputType, args1: Optional[_InputType] = None) -> _OutputType:
        return self._covfun(args0, args1)

    def _sample_at_input(
        self,
        args: _InputType,
        size: ShapeArgType = (),
        random_state: RandomStateArgType = None,
    ) -> _OutputType:
        gaussian_rv = self.__call__(args)
        gaussian_rv.random_state = random_state
        return gaussian_rv.sample(size=size)

    def push_forward(
        self,
        args: _InputType,
        base_measure: Type[randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
