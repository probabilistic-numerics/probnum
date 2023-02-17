"""Markovian processes."""

from typing import Optional, Union

import numpy as np
import scipy.stats

from probnum import functions, randvars, utils
from probnum.randprocs import _random_process, kernels
from probnum.randprocs.markov import _transition, continuous, discrete
from probnum.typing import ShapeLike

InputType = Union[np.floating, np.ndarray]
OutputType = Union[np.floating, np.ndarray]


class _MarkovBase(_random_process.RandomProcess):
    def __init__(
        self,
        *,
        initrv: randvars.RandomVariable,
        transition: _transition.Transition,
        input_shape: ShapeLike = (),
    ):
        self.initrv = initrv
        self.transition = transition

        output_shape = initrv.shape

        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dtype=np.dtype(np.float_),
            mean=functions.LambdaFunction(
                lambda x: self.__call__(args=x).mean,
                input_shape=input_shape,
                output_shape=output_shape,
            ),
            cov=_MarkovBase.Kernel(
                self.__call__,
                input_shape=input_shape,
                output_shape=output_shape,
            ),
        )

    def __call__(self, args: InputType) -> randvars.RandomVariable:
        raise NotImplementedError

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: InputType,
        size: ShapeLike = (),
    ) -> OutputType:

        size = utils.as_shape(size)
        args = np.atleast_1d(args)
        if args.ndim > 1:
            raise ValueError(f"Invalid args shape {args.shape}")

        base_measure_realizations = scipy.stats.norm.rvs(
            size=(size + args.shape + self.initrv.shape), random_state=rng
        )

        if size == ():
            return np.array(
                self.transition.jointly_transform_base_measure_realization_list_forward(
                    base_measure_realizations=base_measure_realizations,
                    t=args,
                    initrv=self.initrv,
                    _diffusion_list=np.ones_like(args[:-1]),
                )
            )

        return np.stack(
            [
                self.transition.jointly_transform_base_measure_realization_list_forward(
                    base_measure_realizations=base_real,
                    t=args,
                    initrv=self.initrv,
                    _diffusion_list=np.ones_like(args[:-1]),
                )
                for base_real in base_measure_realizations
            ]
        )

    class Kernel(kernels.Kernel):
        def __init__(
            self, markov_proc_call, input_shape: ShapeLike, output_shape: ShapeLike
        ):
            self._markov_proc_call = markov_proc_call

            super().__init__(
                input_shape=input_shape,
                output_shape_0=output_shape,
                output_shape_1=output_shape,
            )

        def _evaluate(self, x0: np.ndarray, x1: Optional[np.ndarray]) -> np.ndarray:
            if x1 is None:
                return self._markov_proc_call(args=x0).cov

            raise NotImplementedError


class MarkovProcess(_MarkovBase):
    r"""Random processes with the Markov property.

    A Markov process is a random process with the additional property that
    conditioned on the present state of the system its future and past states are
    independent. This is known as the Markov property or as the process being
    memoryless. A Markov process can be fully defined via an initial state and a
    state transition.

    Parameters
    ----------
    initarg
        Initial starting input of the process.
    initrv
        Random variable describing the initial state.
    transition
        State transition of the system.

    See Also
    --------
    RandomProcess : Random processes.
    GaussianProcess : Gaussian processes.
    """

    def __init__(
        self,
        *,
        initarg: np.ndarray,
        initrv: randvars.RandomVariable,
        transition: continuous.SDE,
    ):
        if not isinstance(transition, continuous.SDE):  # pragma: no cover
            msg = "The transition is not continuous. Did you mean 'MarkovSequence'?"
            raise TypeError(msg)

        super().__init__(
            initrv=initrv,
            transition=transition,
            input_shape=np.asarray(initarg).shape,
        )
        self.initarg = initarg


class MarkovSequence(_MarkovBase):
    """Discrete-time Markov processes."""

    def __init__(
        self,
        *,
        initarg: np.ndarray,
        initrv: randvars.RandomVariable,
        transition: continuous.SDE,
    ):
        if not isinstance(transition, discrete.NonlinearGaussian):  # pragma: no cover
            msg = "The transition is not discrete. Did you mean 'MarkovProcess'?"
            raise TypeError(msg)

        super().__init__(
            initrv=initrv,
            transition=transition,
            input_shape=np.asarray(initarg).shape,
        )
        self.initarg = initarg
