"""Particle filtering posterior."""
import abc
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from probnum import _randomvariablelist
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class ParticleFilterPosterior(FiltSmoothPosterior):
    """Posterior distribution of a particle filter.."""

    # This is essentially just a lightweight wrapper around _RandomVariableList.
    def __init__(
        self, states: _randomvariablelist._RandomVariableList, locations: np.ndarray
    ):
        self.states = _randomvariablelist._RandomVariableList(states)
        self.locations = locations

    def __call__(self, t):
        raise NotImplementedError("Particle filters do not provide dense output.")

    def __len__(self):
        return len(self.states)

    def __getitem__(self):
        return self.states[idx]
