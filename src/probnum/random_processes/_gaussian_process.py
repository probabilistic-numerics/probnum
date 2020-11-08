"""Gaussian processes."""

from typing import Callable, Union

import numpy as np

from probnum.filtsmooth.statespace import DiscreteGaussianLinearModel, LinearSDEModel
from probnum.random_variables import Normal, RandomVariable
from probnum.type import FloatArgType, ShapeArgType

from . import _random_process

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class GaussianProcess(_random_process.RandomProcess[_InputType, _OutputType]):
    """
    Gaussian processes.

    A Gaussian process is a continuous stochastic process which if evaluated at a
    finite set of inputs returns a multivariate normal random variable. Gaussian
    processes are fully characterized by their mean and covariance function.

    Parameters
    ----------
    mean :
        Mean function.
    cov :
        Covariance function or kernel.
    input_shape :
        Shape of the input of the Gaussian process.

    See Also
    --------
    RandomProcess : Class representing random processes.
    GaussMarkovProcess : Gaussian processes with the Markov property.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Gaussian process definition
    >>> mean = lambda x : np.zeros_like(x)  # zero-mean function
    >>> kernel = lambda x0, x1, _: np.exp(
    ...     -0.5 * np.sum(np.subtract.outer(x0, x1) ** 2, axis=(1, 3))
    ... )  # exponentiated quadratic kernel
    >>> gp = GaussianProcess(mean=mean, cov=kernel, input_shape=(), output_shape=())
    >>> # Sample path
    >>> x = np.linspace(-1, 1, 5)[:, None]
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
    ...     mean=mean, cov=cov_coreg_expquad, input_shape=(), output_shape=(2,)
    ... )
    >>> x = np.array([-1, 0, 1])[:, None]
    >>> K = gp.cov(x, x, keepdims=True)
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
        input_shape: ShapeArgType,
        output_shape: ShapeArgType,
        mean: Callable[[_InputType], _OutputType],
        cov: Callable[[_InputType, _InputType, bool], _OutputType],
    ):
        # Type normalization
        # TODO

        # Data type normalization
        # TODO

        # Call to super class
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=np.dtype(np.float_),
            mean=mean,
            cov=cov,
        )

    def __call__(self, x: _InputType) -> Normal:

        # Reshape input to (n, d)
        if len(self.input_shape) == 0:
            x = np.asarray(x).reshape((-1, 1))
        else:
            x = x.reshape((-1,) + self.input_shape)

        return Normal(
            mean=np.squeeze(self.mean(x)), cov=self.cov(x0=x, x1=x, keepdims=False)
        )

    def _sample_at_input(self, x: _InputType, size: ShapeArgType = ()) -> _OutputType:
        rv_at_input = Normal(mean=self.mean(x), cov=self.cov(x0=x, x1=x))
        return rv_at_input.sample(size=size)


class GaussMarkovProcess(GaussianProcess):
    """
    Gaussian processes with the Markov property.

    A Gauss-Markov process is a Gaussian process with the additional property that
    conditioned on the present state of the system its future and past states are
    independent. This is known as the Markov property or as the process being
    memoryless.

    Parameters
    ----------
    linear_transition
        Linear transition model describing a state change of the system.
    initrv
        Initial random variable describing the initial state.
    time0
        Initial starting index / time of the process.

    See Also
    --------
    GaussianProcess : Class representing Gaussian processes.

    Examples
    --------
    """

    def __init__(
        self,
        linear_transition: Union[LinearSDEModel, DiscreteGaussianLinearModel],
        initrv: RandomVariable[_OutputType],
        time0: FloatArgType = 0.0,
    ):
        self.transition = linear_transition
        self.time0 = time0
        self.initrv = initrv
        super().__init__(
            input_shape=(),
            output_shape=initrv.shape,
            mean=self._sde_meanfun,
            cov=self._sde_covfun,
        )

    def _sde_meanfun(self, t):
        return self._transition_rv(t).mean

    def _sde_covfun(self, t0, t1):
        raise NotImplementedError

    def _transition_rv(self, t):
        return self.transition.transition_rv(rv=self.initrv, start=self.time0, stop=t)

    def var(self, x):
        return lambda loc: self._transition_rv(x).cov
