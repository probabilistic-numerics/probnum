"""Importance distributions for particle filtering."""

import abc
from typing import Dict, Tuple

import numpy as np

from probnum.filtsmooth import gaussian
from probnum.randvars import RandomVariable

__all__ = [
    "ImportanceDistribution",
    "BootstrapImportanceDistribution",
    "LinearizationImportanceDistribution",
]


class ImportanceDistribution(abc.ABC):
    """Importance distributions used in particle filtering."""

    def __init__(self, dynamics_model):
        self.dynamics_model = dynamics_model

    @abc.abstractmethod
    def generate_importance_rv(
        self, particle, data, t, dt=None, measurement_model=None
    ) -> Tuple[RandomVariable, RandomVariable, Dict]:
        """Generate an importance distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_initrv_with_data(
        self, initrv, data, t, measurement_model=None
    ) -> Tuple[RandomVariable, RandomVariable, Dict]:
        """Process the initial random variable based on data."""
        raise NotImplementedError

    def log_correction_factor(
        self, proposal_state, importance_rv, dynamics_rv, old_weight
    ) -> float:
        return (
            dynamics_rv.logpdf(proposal_state)
            - importance_rv.logpdf(proposal_state)
            + np.log(old_weight + 1e-14)
        )


class BootstrapImportanceDistribution(ImportanceDistribution):
    """Bootstrap particle filter importance distribution."""

    def generate_importance_rv(
        self, particle, data, t, dt=None, measurement_model=None
    ):
        dynamics_rv, info = self.dynamics_model.forward_realization(
            realization=particle, t=t, dt=dt
        )
        return dynamics_rv, dynamics_rv, info

    def process_initrv_with_data(self, initrv, data, t, measurement_model=None):
        """Process the initial random variable if the initarg is the location of the
        first data point."""
        return initrv, initrv, {}

    def log_correction_factor(
        self, proposal_state, importance_rv, dynamics_rv, old_weight
    ) -> float:
        return 0.0


class LinearizationImportanceDistribution(ImportanceDistribution):
    """Local linearisation importance distribution."""

    def __init__(self, dynamics_model, linearization_strategy):

        # Callable that maps a non-linear model to a Discrete(E/U)KF model
        # To choose either one, consider the class methods below.
        self.linearization_strategy = linearization_strategy
        super().__init__(dynamics_model=dynamics_model)

    @classmethod
    def from_ekf(
        cls,
        dynamics_model,
        forward_implementation="classic",
        backward_implementation="classic",
    ):
        def linearization_strategy(non_linear_model):
            return gaussian.approx.DiscreteEKFComponent(
                non_linear_model,
                forward_implementation=forward_implementation,
                backward_implementation=backward_implementation,
            )

        return cls(
            dynamics_model=dynamics_model, linearization_strategy=linearization_strategy
        )

    @classmethod
    def from_ukf(cls, dynamics_model, spread=1e-4, priorpar=2.0, special_scale=0.0):
        def linearization_strategy(non_linear_model):
            return gaussian.approx.DiscreteUKFComponent(
                non_linear_model,
                spread=spread,
                priorpar=priorpar,
                special_scale=special_scale,
            )

        return cls(
            dynamics_model=dynamics_model, linearization_strategy=linearization_strategy
        )

    def generate_importance_rv(
        self, particle, data, t, dt=None, measurement_model=None
    ):

        if measurement_model is None:
            raise ValueError(
                "Local linearisation importance distributions need a measurement model."
            )
        lin_measmod = self.linearization_strategy(measurement_model)

        info = {}
        dynamics_rv, info["predict_info"] = self.dynamics_model.forward_realization(
            realization=particle, t=t, dt=dt
        )
        importance_rv, info["lin_update_info"] = lin_measmod.backward_realization(
            realization_obtained=data, rv=dynamics_rv, t=t + dt
        )
        return importance_rv, dynamics_rv, info

    def process_initrv_with_data(self, initrv, data, t, measurement_model=None):
        if measurement_model is None:
            raise ValueError(
                "Local linearisation importance distributions need a measurement model."
            )
        lin_measmod = self.linearization_strategy(measurement_model)
        rv, info = lin_measmod.backward_realization(
            realization_obtained=data, rv=initrv, t=t
        )
        return rv, initrv, info
