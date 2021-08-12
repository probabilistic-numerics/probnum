"""Tests for consistencies between IWP, IOUP, and Matern."""


import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.problems.zoo import linalg as linalg_zoo


def both_transitions_matern():
    matern = randprocs.markov.integrator.MaternTransition(
        num_derivatives=2, wiener_process_dimension=2, lengthscale=2.041
    )
    matern2 = randprocs.markov.integrator.MaternTransition(
        num_derivatives=2, wiener_process_dimension=2, lengthscale=2.041
    )
    matern_as_ltisde = randprocs.markov.continuous.LTISDE(
        matern2.drift_matrix, matern2.force_vector, matern2.dispersion_matrix
    )
    return matern, matern_as_ltisde


def both_transitions_ioup():
    ioup = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition(
        num_derivatives=2, wiener_process_dimension=2, driftspeed=2.041
    )
    ioup2 = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition(
        num_derivatives=2, wiener_process_dimension=2, driftspeed=2.041
    )
    ioup_as_ltisde = randprocs.markov.continuous.LTISDE(
        ioup2.drift_matrix, ioup2.force_vector, ioup2.dispersion_matrix
    )
    return ioup, ioup_as_ltisde


def both_transitions_ibm():
    ibm = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=2, wiener_process_dimension=1
    )
    ibm2 = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=2, wiener_process_dimension=1
    )
    ibm_as_ltisde = randprocs.markov.continuous.LTISDE(
        ibm2.drift_matrix, ibm2.force_vector, ibm2.dispersion_matrix
    )
    return ibm, ibm_as_ltisde


@pytest.mark.parametrize(
    "both_transitions",
    [both_transitions_ibm(), both_transitions_ioup(), both_transitions_matern()],
)
def test_same_forward_outputs(both_transitions, diffusion):
    trans1, trans2 = both_transitions
    real = 1 + 0.1 * np.random.rand(trans1.state_dimension)
    out_1, info1 = trans1.forward_realization(
        real, t=0.0, dt=0.5, compute_gain=True, _diffusion=diffusion
    )
    out_2, info2 = trans2.forward_realization(
        real, t=0.0, dt=0.5, compute_gain=True, _diffusion=diffusion
    )
    np.testing.assert_allclose(out_1.mean, out_2.mean)
    np.testing.assert_allclose(out_1.cov, out_2.cov)
    np.testing.assert_allclose(info1["crosscov"], info2["crosscov"])
    np.testing.assert_allclose(info1["gain"], info2["gain"])


@pytest.mark.parametrize(
    "both_transitions",
    [both_transitions_ibm(), both_transitions_ioup(), both_transitions_matern()],
)
def test_same_backward_outputs(both_transitions, diffusion, rng):
    trans1, trans2 = both_transitions
    real = 1 + 0.1 * np.random.rand(trans1.state_dimension)
    real2 = 1 + 0.1 * np.random.rand(trans1.state_dimension)
    cov = linalg_zoo.random_spd_matrix(rng, dim=trans1.state_dimension)
    rv = randvars.Normal(real2, cov)
    out_1, info1 = trans1.backward_realization(
        real, rv, t=0.0, dt=0.5, compute_gain=True, _diffusion=diffusion
    )
    out_2, info2 = trans2.backward_realization(
        real, rv, t=0.0, dt=0.5, compute_gain=True, _diffusion=diffusion
    )
    np.testing.assert_allclose(out_1.mean, out_2.mean)
    np.testing.assert_allclose(out_1.cov, out_2.cov)
