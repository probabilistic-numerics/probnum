"""Particle filters."""

import abc
from dataclasses import dataclass

import numpy as np

from probnum import random_variables
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.filtsmooth.filtsmoothposterior import FiltSmoothPosterior

from ._particle_filter_posterior import ParticleFilterPosterior


def effective_number_of_events(categ_rv):
    return 1.0 / np.sum(categ_rv.event_probabilities ** 2)


def resample_categorical(categ_rv):

    u = np.random.rand(*categ_rv.event_probabilities.shape)
    bins = np.cumsum(categ_rv.event_probabilities)
    new_support = categ_rv.support[np.digitize(u, bins)]

    new_event_probs = np.ones(categ_rv.event_probabilities.shape) / len(
        categ_rv.event_probabilities
    )
    return random_variables.Categorical(
        support=new_support, event_probabilities=new_event_probs
    )


class ParticleFilter(BayesFiltSmooth):
    """Particle filter."""

    def __init__(
        self,
        dynamics_model,
        measurement_model,
        initrv,
        num_particles,
        importance_density_choice="gaussian",
        with_resampling=True,
    ):
        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model
        self.initrv = initrv
        self.num_particles = num_particles
        self.with_resampling = with_resampling

        if importance_density_choice not in ["bootstrap", "gaussian"]:
            raise ValueError
        self.importance_density_choice = importance_density_choice

    def filter(self, dataset, times, _previous_posterior=None):

        # Initialize:
        particles = []
        weights = []
        for idx in range(self.num_particles):

            dynamics_rv = self.initrv
            proposal_state, proposal_weight = self.sequential_importance_sampling(
                dataset[0], times[0], dynamics_rv
            )
            particles.append(proposal_state)
            weights.append(proposal_weight)

        weights = weights / np.sum(weights)
        curr_rv = random_variables.Categorical(
            support=np.array(particles), event_probabilities=weights
        )
        rvs = [curr_rv]

        # Iterate
        for idx in range(1, len(times)):

            curr_rv, _ = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                randvar=curr_rv,
                data=dataset[idx],
            )
            rvs.append(curr_rv)
        return ParticleFilterPosterior(rvs, times)

    def filter_step(self, start, stop, randvar, data):

        new_weights = randvar.event_probabilities.copy()
        new_support = randvar.support.copy()

        for idx, (particle, weight) in enumerate(zip(new_support, new_weights)):

            dynamics_rv, _ = self.dynamics_model.forward_realization(
                particle, t=start, dt=(stop - start)
            )

            proposal_state, proposal_weight = self.sequential_importance_sampling(
                data, stop, dynamics_rv
            )

            new_support[idx] = proposal_state
            new_weights[idx] = proposal_weight

        new_weights = new_weights / np.sum(new_weights)
        new_rv = random_variables.Categorical(
            support=new_support, event_probabilities=new_weights
        )

        # Resample
        if self.with_resampling:
            if effective_number_of_events(new_rv) < self.num_particles / 10:
                new_rv = resample_categorical(new_rv)

        return new_rv, {}

    def sequential_importance_sampling(self, data, stop, dynamics_rv):
        proposal_rv = self.dynamics_to_proposal_rv(dynamics_rv, data=data, t=stop)
        proposal_state = proposal_rv.sample()
        meas_rv, _ = self.measurement_model.forward_realization(proposal_state, t=stop)
        proposal_weight = (
            meas_rv.pdf(data)
            * dynamics_rv.pdf(proposal_state)
            / proposal_rv.pdf(proposal_state)
        )
        return proposal_state, proposal_weight

    def dynamics_to_proposal_rv(self, dynamics_rv, data, t):
        proposal_rv = dynamics_rv
        if self.importance_density_choice == "gaussian":  # "gaussian"
            proposal_rv, _ = self.measurement_model.backward_realization(
                data, proposal_rv, t=t
            )
        return proposal_rv
