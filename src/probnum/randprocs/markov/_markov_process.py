"""Markovian processes."""

from typing import Optional, Type, Union

import numpy as np
import scipy.stats

from probnum import _function, randvars, utils
from probnum.randprocs import _random_process, kernels
from probnum.randprocs.markov import _transition
from probnum.typing import ShapeLike

_InputType = Union[np.floating, np.ndarray]
_OutputType = Union[np.floating, np.ndarray]


class MarkovProcess(_random_process.RandomProcess):
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
        initarg: np.ndarray,
        initrv: randvars.RandomVariable,
        transition: _transition.Transition,
    ):
        self.initarg = initarg
        self.initrv = initrv
        self.transition = transition

        super().__init__(
            input_shape=np.asarray(initarg).shape,
            output_shape=initrv.shape,
            dtype=np.dtype(np.float_),
        )

    def __call__(self, args: _InputType) -> randvars.RandomVariable:
        raise NotImplementedError

    @property
    def mean(self) -> _function.Function:
        return _function.LambdaFunction(
            lambda x: self.__call__(args=x).mean,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
        )

    @property
    def cov(self) -> "MarkovProcess.Kernel":
        return MarkovProcess.Kernel(self.__call__)

    def _sample_at_input(
        self,
        rng: np.random.Generator,
        args: _InputType,
        size: ShapeLike = (),
    ) -> _OutputType:

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

    def push_forward(
        self,
        args: _InputType,
        base_measure: Type[randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    class Kernel(kernels.Kernel):
        def __init__(self, markov_proc: "MarkovProcess"):
            self._randproc_call = markov_proc.__call__

            super().__init__(
                input_shape=markov_proc.input_shape,
                shape=markov_proc.output_shape,
            )

        def _evaluate(
            self, x0: np.ndarray, x1: Optional[np.ndarray]
        ) -> Union[np.ndarray, np.float_]:
            if x1 is None:
                return self._randproc_call(args=x0).cov

            raise NotImplementedError
