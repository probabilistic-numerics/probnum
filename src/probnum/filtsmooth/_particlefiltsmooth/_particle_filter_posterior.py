"""Particle filtering posterior."""

from typing import Optional

import numpy as np

from probnum import randvars
from probnum.typing import (
    DenseOutputLocationArgType,
    FloatArgType,
    RandomStateArgType,
    ShapeArgType,
)

from .._timeseriesposterior import TimeSeriesPosterior


class ParticleFilterPosterior(TimeSeriesPosterior):
    """Posterior distribution of a particle filter.."""

    def __call__(self, t):
        raise NotImplementedError("Particle filters do not provide dense output.")

    # The methods below are not implemented (yet?).

    def interpolate(self, t: FloatArgType) -> randvars.RandomVariable:
        raise NotImplementedError

    def sample(
        self,
        t: Optional[DenseOutputLocationArgType] = None,
        size: Optional[ShapeArgType] = (),
        random_state: Optional[RandomStateArgType] = None,
    ) -> np.ndarray:
        raise NotImplementedError("Sampling is not implemented.")

    def transform_base_measure_realizations(
        self,
        base_measure_realizations: np.ndarray,
        t: Optional[DenseOutputLocationArgType] = None,
    ) -> np.ndarray:
        raise NotImplementedError(
            "Transforming base measure realizations is not implemented."
        )
