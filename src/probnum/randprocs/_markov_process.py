"""Markovian processes."""

from typing import Optional, Type, Union

import numpy as np

from probnum import randvars, statespace
from probnum.type import RandomStateArgType, ShapeArgType

from . import _random_process

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
        transition: statespace.Transition,
    ):
        self.initarg = initarg
        self.initrv = initrv
        self.transition = transition

        super().__init__(
            input_dim=1 if np.asarray(initarg).ndim == 0 else initarg.shape[0],
            output_dim=1 if np.asarray(initrv).ndim == 0 else initrv.shape[0],
            dtype=np.dtype(np.float_),
        )

    def __call__(self, args: _InputType) -> randvars.RandomVariable:
        raise NotImplementedError

    def mean(self, args: _InputType) -> _OutputType:
        return self.__call__(args=args).mean

    def cov(self, args0: _InputType, args1: Optional[_InputType] = None) -> _OutputType:
        if args1 is None:
            return self.__call__(args=args0).cov
        raise NotImplementedError

    def _sample_at_input(
        self,
        args: _InputType,
        size: ShapeArgType = (),
        random_state: RandomStateArgType = None,
    ) -> _OutputType:
        randvar = self.__call__(args=args)
        randvar.random_state = random_state
        return randvar.sample(size=size)

    def push_forward(
        self,
        args: _InputType,
        base_measure: Type[randvars.RandomVariable],
        sample: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError
