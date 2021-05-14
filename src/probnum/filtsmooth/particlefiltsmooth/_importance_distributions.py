"""Importance distributions for particle filtering."""

import abc
from typing import Dict, Tuple

import numpy as np

from probnum.randvars import RandomVariable

__all__ = [
    "ImportanceDistribution",
    "BootstrapImportanceDistribution",
    "LinearizationImportanceDistribution",
]


class ImportanceDistribution(abc.ABC):
    def __init__(self, dynamics_model):
        self.dynamics_model = dynamics_model

    @abc.abstractmethod
    def apply(
        self, particle, data, t, dt=None, lin_measurement_model=None
    ) -> Tuple[RandomVariable, RandomVariable, Dict]:
        """Apply the importance distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def process_initrv_with_data(
        self, initrv, data, t, lin_measurement_model=None
    ) -> Tuple[RandomVariable, RandomVariable, Dict]:
        """Process the initial random variable if the initarg is the location of the
        first data point."""
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

    def apply(self, particle, data, t, dt=None, lin_measurement_model=None):
        dynamics_rv, info = self.dynamics_model.forward_realization(
            realization=particle, t=t, dt=dt
        )
        return dynamics_rv, dynamics_rv, info

    def process_initrv_with_data(self, initrv, data, t, lin_measurement_model=None):
        """Process the initial random variable if the initarg is the location of the
        first data point."""
        return initrv, initrv, {}

    def log_correction_factor(
        self, proposal_state, importance_rv, dynamics_rv, old_weight
    ) -> float:
        return 0.0


class LinearizationImportanceDistribution(ImportanceDistribution):
    """Local linearisation importance distribution."""

    def apply(self, particle, data, t, dt=None, lin_measurement_model=None):
        if lin_measurement_model is None:
            raise ValueError(
                "Local linearisation importance distributions need a linearized measurement model."
            )
        info = {}
        dynamics_rv, info["predict_info"] = self.dynamics_model.forward_realization(
            realization=particle, t=t, dt=dt
        )
        (
            importance_rv,
            info["lin_update_info"],
        ) = lin_measurement_model.backward_realization(
            realization_obtained=data, rv=dynamics_rv, t=t + dt
        )
        return importance_rv, dynamics_rv, info

    def process_initrv_with_data(self, initrv, data, t, lin_measurement_model=None):
        if lin_measurement_model is None:
            raise ValueError(
                "Local linearisation importance distributions need a linearized measurement model."
            )
        rv, info = lin_measurement_model.backward_realization(
            realization_obtained=data, rv=initrv, t=t
        )
        return rv, initrv, info
