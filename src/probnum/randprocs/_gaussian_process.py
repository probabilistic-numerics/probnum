"""Gaussian processes."""

from __future__ import annotations

from probnum import backend, randvars
from probnum.backend.typing import ArrayLike

from . import _random_process, kernels
from .. import functions


class GaussianProcess(_random_process.RandomProcess[ArrayLike, backend.Array]):
    """Gaussian processes.

    A Gaussian process is a continuous stochastic process which if evaluated at a
    finite set of inputs returns a random variable with a normal distribution. Gaussian
    processes are fully characterized by their mean and covariance function.

    Parameters
    ----------
    mean
        Mean function.
    cov
        Covariance function or kernel.

    See Also
    --------
    RandomProcess : Random processes.
    MarkovProcess : Random processes with the Markov property.

    Examples
    --------
    Define a Gaussian process with a zero mean function and RBF kernel.

    >>> from probnum import backend, functions
    >>> from probnum.randprocs.kernels import ExpQuad
    >>> from probnum.randprocs import GaussianProcess
    >>> mu = functions.Zero(input_shape=())
    >>> k = ExpQuad(input_shape=())
    >>> gp = GaussianProcess(mu, k)

    Sample from the Gaussian process.

    >>> x = backend.linspace(-1, 1, 5)
    >>> rng_state = backend.random.rng_state(seed=42)
    >>> gp.sample(rng_state, x)
    array([ 0.30471708, -0.22021158, -0.36160304,  0.05888274,  0.27793918])
    >>> gp.cov.matrix(x)
    array([[1.        , 0.8824969 , 0.60653066, 0.32465247, 0.13533528],
           [0.8824969 , 1.        , 0.8824969 , 0.60653066, 0.32465247],
           [0.60653066, 0.8824969 , 1.        , 0.8824969 , 0.60653066],
           [0.32465247, 0.60653066, 0.8824969 , 1.        , 0.8824969 ],
           [0.13533528, 0.32465247, 0.60653066, 0.8824969 , 1.        ]])
    """

    def __init__(
        self,
        mean: functions.Function,
        cov: kernels.Kernel,
    ):
        if not isinstance(mean, functions.Function):
            raise TypeError("The mean function must have type `probnum.Function`.")

        super().__init__(
            input_shape=mean.input_shape,
            output_shape=mean.output_shape,
            dtype=backend.asdtype(backend.float64),
            mean=mean,
            cov=cov,
        )

    def __call__(self, args: ArrayLike) -> randvars.Normal:
        return randvars.Normal(
            mean=backend.asarray(
                self.mean(args), copy=False  # pylint: disable=not-callable
            ),
            cov=self.cov.matrix(args),
        )
