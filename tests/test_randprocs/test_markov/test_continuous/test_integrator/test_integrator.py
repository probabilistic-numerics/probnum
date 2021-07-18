import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.problems.zoo import linalg as linalg_zoo


class TestIntegratorTransition:
    """An integrator should be usable as is, but its tests are also useful for
    IntegratedWienerProcessTransition, IOUP, etc."""

    # Replacement for an __init__ in the pytest language. See:
    # https://stackoverflow.com/questions/21430900/py-test-skips-test-class-if-constructor-is-defined
    @pytest.fixture(autouse=True)
    def _setup(self, some_nu):
        self.some_nu = some_nu
        self.integrator = randprocs.markov.continuous.integrator.IntegratorTransition(
            nu=self.some_nu, wiener_process_dimension=1
        )

    def test_proj2coord(self):
        base = np.zeros(self.some_nu + 1)
        base[0] = 1
        e_0_expected = np.kron(np.eye(1), base)
        e_0 = self.integrator.proj2coord(coord=0)
        np.testing.assert_allclose(e_0, e_0_expected)

        base = np.zeros(self.some_nu + 1)
        base[-1] = 1
        e_q_expected = np.kron(np.eye(1), base)
        e_q = self.integrator.proj2coord(coord=self.some_nu)
        np.testing.assert_allclose(e_q, e_q_expected)

    def test_precon(self):

        assert isinstance(
            self.integrator.precon,
            randprocs.markov.continuous.integrator.NordsieckLikeCoordinates,
        )


def both_transitions_matern():
    matern = randprocs.markov.continuous.integrator.Matern(
        nu=2, wiener_process_dimension=2, lengthscale=2.041
    )
    matern2 = randprocs.markov.continuous.integrator.Matern(
        nu=2, wiener_process_dimension=2, lengthscale=2.041
    )
    matern_as_ltisde = randprocs.markov.continuous.LTISDE(
        matern2.driftmat, matern2.forcevec, matern2.dispmat
    )
    return matern, matern_as_ltisde


def both_transitions_ioup():
    ioup = randprocs.markov.continuous.integrator.IOUP(
        nu=2, wiener_process_dimension=2, driftspeed=2.041
    )
    ioup2 = randprocs.markov.continuous.integrator.IOUP(
        nu=2, wiener_process_dimension=2, driftspeed=2.041
    )
    ioup_as_ltisde = randprocs.markov.continuous.LTISDE(
        ioup2.driftmat, ioup2.forcevec, ioup2.dispmat
    )
    return ioup, ioup_as_ltisde


def both_transitions_ibm():
    ibm = randprocs.markov.continuous.integrator.IntegratedWienerProcessTransition(
        nu=2, wiener_process_dimension=1
    )
    ibm2 = randprocs.markov.continuous.integrator.IntegratedWienerProcessTransition(
        nu=2, wiener_process_dimension=1
    )
    ibm_as_ltisde = randprocs.markov.continuous.LTISDE(
        ibm2.driftmat, ibm2.forcevec, ibm2.dispmat
    )
    return ibm, ibm_as_ltisde


@pytest.mark.parametrize(
    "both_transitions",
    [both_transitions_ibm(), both_transitions_ioup(), both_transitions_matern()],
)
def test_same_forward_outputs(both_transitions, diffusion):
    trans1, trans2 = both_transitions
    real = 1 + 0.1 * np.random.rand(trans1.dimension)
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
    real = 1 + 0.1 * np.random.rand(trans1.dimension)
    real2 = 1 + 0.1 * np.random.rand(trans1.dimension)
    cov = linalg_zoo.random_spd_matrix(rng, dim=trans1.dimension)
    rv = randvars.Normal(real2, cov)
    out_1, info1 = trans1.backward_realization(
        real, rv, t=0.0, dt=0.5, compute_gain=True, _diffusion=diffusion
    )
    out_2, info2 = trans2.backward_realization(
        real, rv, t=0.0, dt=0.5, compute_gain=True, _diffusion=diffusion
    )
    np.testing.assert_allclose(out_1.mean, out_2.mean)
    np.testing.assert_allclose(out_1.cov, out_2.cov)


############################################################################################
# Tests for the coordinate-representation conversion functions.
############################################################################################


@pytest.fixture
def some_order():
    return 2


@pytest.fixture
def some_dim():
    return 3


@pytest.fixture
def fake_state(some_order, some_dim):
    return np.arange(some_dim * (some_order + 1))


@pytest.fixture
def in_out_pair(fake_state, some_order):
    """Initial states for the three-body initial values.

    Returns (derivwise, coordwise)
    """
    return fake_state, fake_state.reshape((-1, some_order + 1)).T.flatten()


def test_in_out_pair_is_not_identical(in_out_pair):
    """A little sanity check to assert that these are actually different, so the
    conversion is non-trivial."""
    derivwise, coordwise = in_out_pair
    assert np.linalg.norm(derivwise - coordwise) > 5


def test_convert_coordwise_to_derivwise(in_out_pair, some_order, some_dim):
    derivwise, coordwise = in_out_pair
    coordwise_as_derivwise = (
        randprocs.markov.continuous.integrator.utils.convert_coordwise_to_derivwise(
            coordwise, some_order, some_dim
        )
    )
    np.testing.assert_allclose(coordwise_as_derivwise, derivwise)


def test_convert_derivwise_to_coordwise(in_out_pair, some_order, some_dim):
    derivwise, coordwise = in_out_pair
    derivwise_as_coordwise = (
        randprocs.markov.continuous.integrator.utils.convert_derivwise_to_coordwise(
            derivwise, some_order, some_dim
        )
    )
    np.testing.assert_allclose(derivwise_as_coordwise, coordwise)


def test_conversion_pairwise_inverse(in_out_pair, some_order, some_dim):
    derivwise, coordwise = in_out_pair
    as_coord = (
        randprocs.markov.continuous.integrator.utils.convert_derivwise_to_coordwise(
            derivwise, some_order, some_dim
        )
    )
    as_deriv_again = (
        randprocs.markov.continuous.integrator.utils.convert_coordwise_to_derivwise(
            as_coord, some_order, some_dim
        )
    )
    np.testing.assert_allclose(as_deriv_again, derivwise)
