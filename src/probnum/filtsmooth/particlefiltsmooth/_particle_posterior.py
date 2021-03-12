"""Particle filtering posterior."""
import abc
from dataclasses import dataclass

import numpy as np

from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior


class ParticleFilterPosterior(FiltSmoothPosterior):
    def __init__(self, particle_state_list, locations):
        self.particle_state_list = particle_state_list
        self.locations = locations

    def __call__(self, t):
        raise NotImplementedError

    def __len__(self):
        return len(self.particle_state_list)

    def __getitem__(self):
        state = self.particle_state_list[idx]
        return state.particles, state.weights
