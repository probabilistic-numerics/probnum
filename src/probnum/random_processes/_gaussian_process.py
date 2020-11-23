"""Gaussian processes."""

from typing import Callable, Union

import numpy as np

import probnum.kernels as kernels
import probnum.utils as _utils
from probnum.random_variables import Normal
from probnum.type import IntArgType, RandomStateArgType, ShapeArgType

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
    input_dim :
        Shape of the input of the Gaussian process.
    output_dim :
        Shape of the output of the Gaussian process.
    mean :
        Mean function.
    cov :
        Covariance function or kernel.
    random_state :
        Random state of the random process. If None (or np.random), the global
        :mod:`numpy.random` state is used. If integer, it is used to seed the local
        :class:`~numpy.random.RandomState` instance.

    See Also
    --------
    RandomProcess : Class representing random processes.
    GaussMarkovProcess : Gaussian processes with the Markov property.

    Examples
    --------
    >>> import numpy as np
    >>> from probnum.random_processes import GaussianProcess
    >>> # Gaussian process definition
    >>> mean = lambda x : np.zeros_like(x)  # zero-mean function
    >>> kernel = lambda x0, x1, _: np.exp(
    ...     -0.5 * np.sum(np.subtract.outer(x0, x1) ** 2, axis=(1, 3))
    ... )  # exponentiated quadratic kernel
    >>> gp = GaussianProcess(mean=mean, cov=kernel)
    >>> # Sample path
    >>> x = np.linspace(-1, 1, 5)[:, None]
    >>> np.random.seed(42)
    >>> gp.sample(x)
    array([[-0.35187364],
           [-0.41301096],
           [-0.65094306],
           [-0.56817194],
           [ 0.01173088]])

    >>> # Multi-output Gaussian process
    >>> cov_coreg_expquad = lambda x0, x1, _: np.multiply.outer(
    ...     kernel(x0, x1, _), np.array([[4, 2], [2, 1]])
    ... )
    >>> gp = GaussianProcess(
    ...     mean=mean, cov=cov_coreg_expquad, output_shape=2
    ... )
    >>> x = np.array([-1, 0, 1])[:, None]
    >>> K = gp.cov(x)
    >>> K.shape
    (3, 3, 2, 2)
    >>> # Covariance matrix in output-dimension-first order
    >>> np.transpose(K, axes=[2, 0, 3, 1]).reshape(2 * x.shape[0], 2 * x.shape[0])
    array([[4.        , 2.42612264, 0.54134113, 2.        , 1.21306132,
            0.27067057],
           [2.42612264, 4.        , 2.42612264, 1.21306132, 2.        ,
            1.21306132],
           [0.54134113, 2.42612264, 4.        , 0.27067057, 1.21306132,
            2.        ],
           [2.        , 1.21306132, 0.27067057, 1.        , 0.60653066,
            0.13533528],
           [1.21306132, 2.        , 1.21306132, 0.60653066, 1.        ,
            0.60653066],
           [0.27067057, 1.21306132, 2.        , 0.13533528, 0.60653066,
            1.        ]])
    """

    def __init__(
        self,
        mean: Callable[[_InputType], _OutputType],
        cov: Union[Callable[[_InputType], _OutputType], kernels.Kernel],
        input_dim: IntArgType = 1,
        output_dim: IntArgType = 1,
        random_state: RandomStateArgType = None,
    ):

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=np.dtype(np.float_),
            random_state=random_state,
            mean=mean,
            cov=cov,
            sample_at_input=self._sample_at_input,
        )

    def __call__(self, x: _InputType) -> Normal:
        x = np.asarray(x)
        if x.ndim == 0:
            mean_eval = _utils.as_numpy_scalar(self.mean(x).squeeze())
            cov_eval = _utils.as_numpy_scalar(self.cov(x).squeeze())
        elif x.ndim == 1:
            mean_eval = self.mean(x).reshape(self.output_dim)
            cov_eval = self.cov(x)
        else:
            mean_eval = self.mean(x).reshape(x.shape[0], self.output_dim)
            cov_eval = self.cov(x)

        return Normal(mean=mean_eval, cov=cov_eval)

    def _sample_at_input(self, x: _InputType, size: ShapeArgType = ()) -> _OutputType:
        return self.__call__(x).sample(size=size)
