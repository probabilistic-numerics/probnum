"""Particle filters."""

import itertools
from collections import abc
from typing import Iterable, Optional, Union

import numpy as np

from probnum import problems, randvars, statespace
from probnum.filtsmooth.bayesfiltsmooth import BayesFiltSmooth
from probnum.type import FloatArgType, IntArgType

from ._particle_filter_posterior import ParticleFilterPosterior

# Terribly long variable names, but internal only, so no worries.
ParticleFilterMeasurementModelArgType = Union[
    statespace.DiscreteGaussian, Iterable[statespace.DiscreteGaussian]
]
ParticleFilterLinearisedMeasurementModelArgType = Union[
    statespace.DiscreteGaussian, Iterable[statespace.DiscreteGaussian]
]


def effective_number_of_events(categ_rv: randvars.Categorical) -> float:
    """Approximate effective number of events in the support of a categorical random
    variable.

    In a particle filter, this is used as the effective number of
    particles which may indicate the need for resampling.
    """
    return 1.0 / np.sum(categ_rv.probabilities ** 2)


class ParticleFilter(BayesFiltSmooth):
    r"""Particle filter (PF). Also known as sequential Monte Carlo method.

    A PF estimates the posterior distribution of a Markov process given noisy, non-linear observations,
    with a set of particles.

    The random state of the particle filter is inferred from the random state of the initial random variable.

    Parameters
    ----------
    dynamics_model :
        Prior dynamics. Since the PF is essentially a discrete-time algorithm,
        the prior must be a discrete model (or at least one with an equivalent discretisation).
        This transition must support `forward_realization`.
    initrv :
        Initial random variable. Can be any `RandomVariable` object that implements `sample()`.
    num_particles :
        Number of particles to use.
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
        dynamics_model: Union[statespace.LTISDE, statespace.DiscreteGaussian],
        initrv: randvars.RandomVariable,
        num_particles: IntArgType,
        with_resampling: bool = True,
        resampling_percentage_threshold: FloatArgType = 0.1,
    ) -> None:
        super().__init__(
            dynamics_model=dynamics_model,
            initrv=initrv,
        )
        self.num_particles = num_particles

        self.with_resampling = with_resampling
        self.resampling_percentage_threshold = resampling_percentage_threshold
        self.min_effective_num_of_particles = (
            resampling_percentage_threshold * num_particles
        )

        # # If None, the dynamics model is used as a fallback option
        # # which results in the bootstrap PF.
        # # Any linearised measurement model that could be used in a
        # # Gaussian filter can be used here and will likely be a better
        # # choice than the bootstrap.
        # self.linearized_measurement_model = linearized_measurement_model

    def filter(
        self,
        regression_problem: problems.RegressionProblem,
        measurement_model: ParticleFilterMeasurementModelArgType,
        linearized_measurement_model: Optional[
            ParticleFilterLinearisedMeasurementModelArgType
        ] = None,
    ):
        """Apply particle filtering to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
        measurement_model :
            Measurement model. Must be a discrete model that supports `forward_realization`.
        linearized_measurement_model :
            Linearized measurement model that is used as an importance density. In principle,
            any discrete-time model that supports `backward_realization` is applicable.
            In practice, it will almost always be one out of `DiscreteEKFComponent`, `DiscreteUKFComponent`,
            or `IteratedDiscreteComponent`. Linear components are also possible, but would most often imply
            that a particle filter is not required, because the filtering problem can be used much faster
            with a Kalman filter. The exception to this rule is if the initial random variable is not Gaussian.
            Optional. Default is None, which implies the bootstrap PF.

        Returns
        -------
        posterior
            Posterior distribution of the filtered output
        info_dicts
            list of dictionaries containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """
        filtered_rvs = []
        info_dicts = []

        for rv, info in self.filter_generator(
            regression_problem, measurement_model, linearized_measurement_model
        ):
            filtered_rvs.append(rv)
            info_dicts.append(info)

        posterior = ParticleFilterPosterior(
            states=filtered_rvs,
            locations=regression_problem.locations,
        )

        return posterior, info_dicts

    def filter_generator(
        self,
        regression_problem: problems.RegressionProblem,
        measurement_model: ParticleFilterMeasurementModelArgType,
        linearized_measurement_model: Optional[
            ParticleFilterLinearisedMeasurementModelArgType
        ] = None,
    ):
        """Apply Particle filtering to a data set.

        Parameters
        ----------
        regression_problem :
            Regression problem.
        measurement_model :
            Measurement model. Must be a discrete model that supports `forward_realization`.
        linearized_measurement_model :
            Linearized measurement model that is used as an importance density. In principle,
            any discrete-time model that supports `backward_realization` is applicable.
            In practice, it will almost always be one out of `DiscreteEKFComponent`, `DiscreteUKFComponent`,
            or `IteratedDiscreteComponent`. Linear components are also possible, but would most often imply
            that a particle filter is not required, because the filtering problem can be used much faster
            with a Kalman filter. The exception to this rule is if the initial random variable is not Gaussian.
            Optional. Default is None, which implies the bootstrap PF.

        Yields
        ------
        curr_rv
            Filtering random variable at each grid point.
        info_dict
            Dictionary containing filtering information

        See Also
        --------
        RegressionProblem: a regression problem data class
        """

        dataset, times = regression_problem.observations, regression_problem.locations

        # It is not clear to me at the moment how to handle this.
        if not np.all(np.diff(times) > 0):
            raise ValueError(
                "Particle filtering cannot handle repeating time points currently."
            )

        if not isinstance(measurement_model, abc.Iterable):
            measurement_model = itertools.repeat(measurement_model, len(times))
        if not isinstance(linearized_measurement_model, abc.Iterable):
            linearized_measurement_model = itertools.repeat(
                linearized_measurement_model, len(times)
            )

        t_old = times[0]
        particles = self.initrv.sample(size=self.num_particles)
        weights = np.ones(self.num_particles) / self.num_particles
        curr_rv = randvars.Categorical(
            support=particles, probabilities=weights, random_state=self.random_state
        )

        # Iterate over data and measurement models
        for t, data, measmod, lin_measmod in itertools.zip_longest(
            times,
            dataset,
            measurement_model,
            linearized_measurement_model,
            fillvalue="None",
        ):
            if (
                t == "None"
                or data == "None"
                or measmod == "None"
                or lin_measmod == "None"
            ):
                errormsg = (
                    "The lengths of the dataset, times and"
                    "measurement models are inconsistent."
                )
                raise ValueError(errormsg)

            dt = t - t_old
            info_dict = {}

            if dt > 0:
                curr_rv, info_dict = self.filter_step(
                    start=t_old,
                    stop=t,
                    randvar=curr_rv,
                    data=data,
                    measurement_model=measmod,
                    linearized_measurement_model=lin_measmod,
                )
                yield curr_rv, info_dict

        ################################
        ################################
        ################################
        ################################
        particles_and_weights = np.array(
            [
                self.compute_new_particle(dataset[0], times[0], self.initrv)
                for _ in range(self.num_particles)
            ],
            dtype=object,
        )
        particles = np.stack(particles_and_weights[:, 0], axis=0)
        weights = np.stack(particles_and_weights[:, 1], axis=0)
        weights = np.array(weights) / np.sum(weights)
        curr_rv = randvars.Categorical(
            support=particles, probabilities=weights, random_state=self.random_state
        )
        yield curr_rv, {}

        for idx in range(1, len(times)):
            curr_rv, info_dict = self.filter_step(
                start=times[idx - 1],
                stop=times[idx],
                randvar=curr_rv,
                data=dataset[idx],
                measurement_model=measmod,
                linearized_measurement_model=lin_measmod,
            )
            yield curr_rv, info_dict

    def filter_step(self, start, stop, randvar, data, measmod, lin_measmod):
        """Perform a particle filter step.

        This method implements sequential importance (re)sampling.

        It consists of the following steps:
        1. Propagating the "past" particles through the dynamics model.
        2. Computing a "proposal" random variable.
        This is either the prior dynamics model or the output of a filter step
        of an (approximate) Gaussian filter.
        3. Sample from the proposal random variable. This is the "new" particle.
        4. Propagate the particle through the measurement model.
        This is required in order to evaluate the PDF of the resulting RV at
        the data. If this is small, the weight of the particle will be small.
        5. Compute weights ("event probabilities") of the new particle.
        This requires evaluating the PDFs of all three RVs (dynamics, proposal, measurement).

        After this is done for all particles, the weights are normalized in order to sum to 1.
        If the effective number of particles is low, the particles are resampled.
        """
        new_weights = randvar.probabilities.copy()
        new_support = randvar.support.copy()

        for idx, (particle, weight) in enumerate(zip(new_support, new_weights)):

            dynamics_rv, _ = self.dynamics_model.forward_realization(
                particle, t=start, dt=(stop - start)
            )
            proposal_state, proposal_weight = self.compute_new_particle(
                data, stop, dynamics_rv, measmod, lin_measmod
            )
            new_support[idx] = proposal_state
            new_weights[idx] = proposal_weight

        new_weights = new_weights / np.sum(new_weights)
        new_rv = randvars.Categorical(
            support=new_support,
            probabilities=new_weights,
            random_state=self.random_state,
        )

        if self.with_resampling:
            if effective_number_of_events(new_rv) < self.min_effective_num_of_particles:
                new_rv = new_rv.resample()

        return new_rv, {}

    def compute_new_particle(self, data, stop, dynamics_rv, measmod, lin_measmod):
        """Compute a new particle.

        Turn the dynamics RV into a proposal RV, apply the measurement
        model and compute new weights via the respective PDFs.
        """
        proposal_rv = self.dynamics_to_proposal_rv(
            dynamics_rv, data=data, t=stop, measmod=measmod, lin_measmod=lin_measmod
        )
        proposal_state = proposal_rv.sample()
        meas_rv, _ = measmod.forward_realization(proposal_state, t=stop)

        # For the bootstrap PF, the dynamics and proposal PDFs cancel out.
        # Therefore we make the following exception.
        if lin_measmod is None:
            log_proposal_weight = meas_rv.logpdf(data)
        else:
            log_proposal_weight = (
                meas_rv.logpdf(data)
                + dynamics_rv.logpdf(proposal_state)
                - proposal_rv.logpdf(proposal_state)
            )

        proposal_weight = np.exp(log_proposal_weight)
        return proposal_state, proposal_weight

    def dynamics_to_proposal_rv(self, dynamics_rv, data, t, measmod, lin_measmod):
        """Turn a dynamics RV into a proposal RV.

        The output of this function depends on the choice of PF. For the
        bootstrap PF, nothing happens. For other PFs, the importance
        density is used to improve the proposal. Currently, only
        approximate Gaussian importance densities are provided.
        """
        proposal_rv = dynamics_rv
        if lin_measmod is not None:
            proposal_rv, _ = lin_measmod.backward_realization(data, proposal_rv, t=t)
        return proposal_rv

    @property
    def random_state(self):
        """Random state of the particle filter.

        Inferred from the random state of the initial random variable.
        """
        return self.initrv.random_state
