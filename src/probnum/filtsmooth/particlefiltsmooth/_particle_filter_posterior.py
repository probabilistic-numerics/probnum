"""Particle filtering posterior."""

import numpy as np

from probnum import _randomvariablelist
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class ParticleFilterPosterior(FiltSmoothPosterior):
    """Posterior distribution of a particle filter.."""

    # This is essentially just a lightweight wrapper around _RandomVariableList.
    def __init__(
        self, states: _randomvariablelist._RandomVariableList, locations: np.ndarray
    ):
        self.states = _randomvariablelist._RandomVariableList(states)
        self.locations = locations
        super().__init__()

    def __call__(self, t):
        raise NotImplementedError("Particle filters do not provide dense output.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]
