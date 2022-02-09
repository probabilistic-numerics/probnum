"""Gaussian processes."""

from typing import Callable, Optional, Type, Union

import numpy as np

from probnum import randvars
from probnum.typing import ShapeLike

from . import _random_process, kernels

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
    >>> from probnum.randprocs.kernels import ExpQuad
    >>> from probnum.randprocs import GaussianProcess
    >>> mu = lambda x : np.zeros_like(x)  # zero-mean function
    >>> k = ExpQuad(input_dim=1)  # RBF kernel
    >>> gp = GaussianProcess(mu, k)

    Sample from the Gaussian process.

    >>> x = np.linspace(-1, 1, 5)[:, None]
    >>> rng = np.random.default_rng(seed=42)
    >>> gp.sample(rng, x)
    array([[-0.7539949 ],
           [-0.6658092 ],
           [-0.52972512],
           [ 0.0674298 ],
           [ 0.72066223]])
    >>> gp.covmatrix(x)
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

        if cov.shape != () and not (
            len(cov.shape) == 2 and cov.shape[0] == cov.shape[1]
        ):
            raise ValueError(
                "Only kernels with shape `()` or `(D, D)` are allowed as covariance "
                "functions of random processes."
            )

        self._meanfun = mean
        self._covfun = cov
        super().__init__(
            input_dim=cov.input_dim,
            output_dim=None if cov.shape == () else cov.shape[0],
            dtype=np.dtype(np.float_),
        )

    def __call__(self, args: _InputType) -> randvars.Normal:
        return randvars.Normal(
            mean=np.array(self.mean(args), copy=False), cov=self.covmatrix(args)
        )

    def mean(self, args: _InputType) -> _OutputType:
        return self._meanfun(args)

    def cov(self, args0: _InputType, args1: Optional[_InputType] = None) -> _OutputType:
        return self._covfun(args0, args1)

    def covmatrix(
        self, args0: _InputType, args1: Optional[_InputType] = None
    ) -> _OutputType:
        return self._covfun.matrix(args0, args1)

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: _InputType,
        size: ShapeLike = (),
    ) -> _OutputType:
        gaussian_rv = self.__call__(args)
        return gaussian_rv.sample(rng=rng, size=size)

    def push_forward(
        self,
        args: _InputType,
        base_measure: Type[randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
