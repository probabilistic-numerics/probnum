"""Particle filters."""

from typing import Iterable, Union

import numpy as np

from probnum import problems, randprocs, randvars
from probnum.filtsmooth import _bayesfiltsmooth
from probnum.filtsmooth.particle import (
    _importance_distributions,
    _particle_filter_posterior,
)
from probnum.typing import FloatArgType, IntArgType

# Terribly long variable names, but internal only, so no worries.
ParticleFilterMeasurementModelArgType = Union[
    randprocs.markov.discrete.DiscreteGaussian,
    Iterable[randprocs.markov.discrete.DiscreteGaussian],
]
ParticleFilterLinearisedMeasurementModelArgType = Union[
    randprocs.markov.discrete.DiscreteGaussian,
    Iterable[randprocs.markov.discrete.DiscreteGaussian],
]


def effective_number_of_events(categ_rv: randvars.Categorical) -> float:
    """Approximate effective number of events in the support of a categorical random
    variable.

    In a particle filter, this is used as the effective number of
    particles which may indicate the need for resampling.
    """
    return 1.0 / np.sum(categ_rv.probabilities ** 2)


class ParticleFilter(_bayesfiltsmooth.BayesFiltSmooth):
    r"""Particle filter (PF). Also known as sequential Monte Carlo method.

    A PF estimates the posterior distribution of a Markov process given noisy, non-linear observations,
    with a set of particles.

    The random state of the particle filter is inferred from the random state of the initial random variable.

    Parameters
    ----------
    prior_process :
        Prior Gauss-Markov process.
    importance_distribution :
        Importance distribution.
    num_particles :
        Number of particles to use.
    rng :
        Random number generator.
    with_resampling :
        Whether after each step the effective number of particles shall be checked, and, if too low,
        the state should be resampled. Optional. Default is `True`.
    resampling_percentage_threshold :
        Percentage threshold for resampling. That is, it is the value :math:`p` such that
        resampling is performed if :math:`N_{\text{eff}} < p \, N_\text{particles}` holds.
        Optional. Default is 0.1. If this value is non-positive, resampling is never performed.
        If it is larger than 1, resampling is performed after each step.
    """

    def __init__(
        self,
        prior_process: randprocs.markov.MarkovProcess,
        importance_distribution: _importance_distributions.ImportanceDistribution,
        num_particles: IntArgType,
        rng: np.random.Generator,
        with_resampling: bool = True,
        resampling_percentage_threshold: FloatArgType = 0.1,
    ) -> None:
        super().__init__(
            prior_process=prior_process,
        )
        self.num_particles = num_particles
        self.importance_distribution = importance_distribution
        self.rng = rng

        self.with_resampling = with_resampling
        self.resampling_percentage_threshold = resampling_percentage_threshold
        self.min_effective_num_of_particles = (
            resampling_percentage_threshold * num_particles
        )

    def filter(
        self,
        regression_problem: problems.TimeSeriesRegressionProblem,
    ):
        """Apply particle filtering to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.

        Returns
        -------
        posterior
            Posterior distribution of the filtered output
        info_dicts
            list of dictionaries containing filtering information

        See Also
        --------
        TimeSeriesRegressionProblem: a regression problem data class
        """
        filtered_rvs = []
        info_dicts = []

        for rv, info in self.filter_generator(regression_problem):
            filtered_rvs.append(rv)
            info_dicts.append(info)

        posterior = _particle_filter_posterior.ParticleFilterPosterior(
            states=filtered_rvs,
            locations=regression_problem.locations,
        )

        return posterior, info_dicts

    def filter_generator(
        self,
        regression_problem: problems.TimeSeriesRegressionProblem,
    ):
        """Apply Particle filtering to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.

        Yields
        ------
        curr_rv
            Filtering random variable at each grid point.
        info_dict
            Dictionary containing filtering information

        See Also
        --------
        TimeSeriesRegressionProblem: a regression problem data class
        """

        # It is not clear at the moment how to handle this.
        if not np.all(np.diff(regression_problem.locations) > 0):
            raise ValueError(
                "Particle filtering cannot handle repeating time points currently."
            )

        initarg = self.prior_process.initarg
        t_old = self.prior_process.initarg

        # If the initial time of the prior equals the location of the first data point,
        # the initial set of particles is overwritten. Here, we set them to unimportant values.
        # If the initial time of the prior is NOT the location of the first data point,
        # we have to sample an initial set of particles.
        weights = np.ones(self.num_particles) / self.num_particles
        particle_set_shape = (self.num_particles,) + self.prior_process.initrv.shape
        if regression_problem.locations[0] == initarg:
            particles = np.nan * np.ones(particle_set_shape)
        else:
            particles = self.prior_process.initrv.sample(
                rng=self.rng, size=(self.num_particles,)
            )

        for t, data, measmod in regression_problem:

            dt = t - t_old
            new_particles = particles.copy()
            new_weights = weights.copy()

            # Capture the inputs in a variable for more compact code layout
            inputs = measmod, particles, weights, data, t_old, dt, t
            if t == initarg:
                particle_generator = self.importance_rv_generator_initial_time(*inputs)
            else:
                particle_generator = self.importance_rv_generator(*inputs)

            for idx, (importance_rv, dynamics_rv, p, w) in enumerate(
                particle_generator
            ):

                # Importance sampling step
                new_particle = importance_rv.sample(rng=self.rng)
                meas_rv, _ = measmod.forward_realization(new_particle, t=t)
                loglikelihood = meas_rv.logpdf(data)
                log_correction_factor = (
                    self.importance_distribution.log_correction_factor(
                        proposal_state=new_particle,
                        importance_rv=importance_rv,
                        dynamics_rv=dynamics_rv,
                        old_weight=w,
                    )
                )
                new_weight = np.exp(loglikelihood + log_correction_factor)

                new_particles[idx] = new_particle
                new_weights[idx] = new_weight

            weights = new_weights / np.sum(new_weights)
            particles = new_particles
            new_rv = randvars.Categorical(support=particles, probabilities=weights)

            if self.with_resampling:
                N = effective_number_of_events(new_rv)
                if N < self.min_effective_num_of_particles:
                    new_rv = new_rv.resample(rng=self.rng)
            yield new_rv, {}
            t_old = t

    def importance_rv_generator_initial_time(
        self,
        measmod,
        particles,
        weights,
        data,
        t_old,
        dt,
        t,
    ):

        processed = self.importance_distribution.process_initrv_with_data(
            self.prior_process.initrv, data, t, measurement_model=measmod
        )
        importance_rv, dynamics_rv, _ = processed
        for p, w in zip(particles, weights):
            yield importance_rv, dynamics_rv, p, w

    def importance_rv_generator(
        self,
        measmod,
        particles,
        weights,
        data,
        t_old,
        dt,
        t,
    ):

        for p, w in zip(particles, weights):
            output = self.importance_distribution.generate_importance_rv(
                p, data, t_old, dt, measurement_model=measmod
            )
            importance_rv, dynamics_rv, _ = output
            yield importance_rv, dynamics_rv, p, w
