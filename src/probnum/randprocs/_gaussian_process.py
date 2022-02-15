"""Gaussian processes."""

from typing import Type, Union

import numpy as np

from probnum import randvars
from probnum.typing import ShapeLike

from . import _random_process, kernels
from .. import _function

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
    >>> from probnum.randprocs.mean_fns import Zero
    >>> from probnum.randprocs.kernels import ExpQuad
    >>> from probnum.randprocs import GaussianProcess
    >>> mu = Zero(input_shape=())  # zero-mean function
    >>> k = ExpQuad(input_shape=())  # RBF kernel
    >>> gp = GaussianProcess(mu, k)

    Sample from the Gaussian process.

    >>> x = np.linspace(-1, 1, 5)
    >>> rng = np.random.default_rng(seed=42)
    >>> gp.sample(rng, x)
    array([-0.7539949 , -0.6658092 , -0.52972512,  0.0674298 ,  0.72066223])
    >>> gp.cov.matrix(x)
    array([[1.        , 0.8824969 , 0.60653066, 0.32465247, 0.13533528],
           [0.8824969 , 1.        , 0.8824969 , 0.60653066, 0.32465247],
           [0.60653066, 0.8824969 , 1.        , 0.8824969 , 0.60653066],
           [0.32465247, 0.60653066, 0.8824969 , 1.        , 0.8824969 ],
           [0.13533528, 0.32465247, 0.60653066, 0.8824969 , 1.        ]])
    """

    def __init__(
        self,
        mean: _function.Function,
        cov: kernels.Kernel,
    ):
        if not isinstance(mean, _function.Function):
            raise TypeError("The mean function must have type `probnum.Function`.")

        if not isinstance(cov, kernels.Kernel):
            raise TypeError(
                "The covariance functions must be implemented as a " "`Kernel`."
            )

        if len(mean.input_shape) > 1:
            raise ValueError(
                "The mean function must have input shape `()` or `(D_in,)`."
            )

        if len(mean.output_shape) > 1:
            raise ValueError(
                "The mean function must have output shape `()` or `(D_out,)`."
            )

        if mean.input_shape != cov.input_shape:
            raise ValueError(
                "The mean and covariance functions must have the same input shapes "
                f"(`mean.input_shape` is {mean.input_shape} and `cov.input_shape` is "
                f"{cov.input_shape})."
            )

        if 2 * mean.output_shape != cov.shape:
            raise ValueError(
                "The shape of the `Kernel` must be a tuple of the form "
                "`(output_shape, output_shape)`, where `output_shape` is the output "
                "shape of the mean function."
            )

        self._mean = mean
        self._cov = cov

        super().__init__(
            input_shape=mean.input_shape,
            output_shape=mean.output_shape,
            dtype=np.dtype(np.float_),
        )

    def __call__(self, args: _InputType) -> randvars.Normal:
        return randvars.Normal(
            mean=np.array(self.mean(args), copy=False), cov=self.cov.matrix(args)
        )

    @property
    def mean(self) -> _function.Function:
        return self._mean

    @property
    def cov(self) -> kernels.Kernel:
        return self._cov

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
