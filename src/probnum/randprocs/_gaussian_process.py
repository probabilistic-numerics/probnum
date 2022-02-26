"""Gaussian processes."""
import numpy as np

from probnum import randvars

from . import _random_process, kernels
from .. import _function
from ..typing import ArrayLike


class GaussianProcess(_random_process.RandomProcess[ArrayLike, np.ndarray]):
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

        super().__init__(
            input_shape=mean.input_shape,
            output_shape=mean.output_shape,
            dtype=np.dtype(np.float_),
            mean=mean,
            cov=cov,
        )

    def __call__(self, args: ArrayLike) -> randvars.Normal:
        return randvars.Normal(
            mean=np.array(self.mean(args), copy=False),  # pylint: disable=not-callable
            cov=self.cov.matrix(args),
        )
