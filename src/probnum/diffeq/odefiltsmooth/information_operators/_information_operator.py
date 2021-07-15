"""Interface for information operators."""

import abc
import dataclasses

import numpy as np

from probnum import problems, statespace
from probnum.typing import FloatArgType

__all__ = ["ODEInformation", "InformationOperator", "FirstOrderODEResidual"]


@dataclasses.dataclass
class ODEInformation:
    """ODE Information used in a probabilistic solver."""

    information_model: statespace.DiscreteGaussian
    ivp: problems.InitialValueProblem

    h0: np.ndarray
    h1: np.ndarray


class InformationOperator(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self, ivp: problems.InitialValueProblem, prior: statespace.Integrator
    ) -> ODEInformation:

        raise NotImplementedError


class FirstOrderODEResidual:
    def __call__(
        self, ivp: problems.InitialValueProblem, prior_transition: statespace.Integrator
    ) -> ODEInformation:

        h0 = prior_transition.proj2coord(coord=0)
        h1 = prior_transition.proj2coord(coord=1)

        def dyna(t, x):
            return h1 @ x - ivp.f(t, h0 @ x)

        def jacobian(t, x):
            return h1 - ivp.df(t, h0 @ x) @ h0

        input_dim = prior_transition.dimension
        output_dim = ivp.dimension
        info_model = statespace.DiscreteGaussian.from_callable(
            input_dim=input_dim,
            output_dim=output_dim,
            state_trans_fun=dyna,
            jacob_state_trans_fun=jacobian,
        )

        return ODEInformation(information_model=info_model, ivp=ivp, h0=h0, h1=h1)
