"""Gaussian processes."""

from typing import Callable, Union

import numpy as np

from probnum import kernels, randvars
from probnum.type import IntArgType, RandomStateArgType, ShapeArgType

from ._random_process import RandomProcess

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussianProcess(RandomProcess[_InputType, _OutputType]):
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

    See Also
    --------
    RandomProcess : Random processes.
    GaussMarkovProcess : Gaussian processes with the Markov property.

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
        cov: Union[Callable[[_InputType], _OutputType], kernels.Kernel],
        input_dim: IntArgType = None,
        output_dim: IntArgType = None,
    ):

        if isinstance(cov, kernels.Kernel):
            input_dim = cov.input_dim
            output_dim = cov.output_dim

        if input_dim is None or output_dim is None:
            raise ValueError(
                "If 'cov' is not a Kernel, 'input_dim' and 'output_dim' must be "
                "specified."
            )
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=np.dtype(np.float_),
            mean=mean,
            cov=cov,
            sample_at_input=self._sample_at_input,
        )

    def __call__(self, x: _InputType) -> randvars.Normal:
        return randvars.Normal(mean=self.mean(x), cov=self.cov(x))

    def _sample_at_input(
        self,
        x: _InputType,
        size: ShapeArgType = (),
        random_state: RandomStateArgType = None,
    ) -> _OutputType:
        gaussian_rv = self.__call__(x)
        gaussian_rv.random_state = random_state
        return gaussian_rv.sample(size=size)
