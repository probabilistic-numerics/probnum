import numpy as np
import pytest

from probnum import randprocs, randvars
from probnum.problems.zoo import linalg as linalg_zoo


class IntegratorMixInTestMixIn:
    def test_uses_mixin(self):
        assert isinstance(self.transition, randprocs.markov.integrator.IntegratorMixIn)

    def test_num_derivatives(self):

        # Check for > 0, because we do not know how many the test cases generate.
        # If this property does not exist, however, this test will fail.
        assert self.transition.num_derivatives > 0

    #
    # def test_proj2coord(self):
    #     base = np.zeros(self.transition.num_derivatives + 1)
    #     base[0] = 1
    #     e_0_expected = np.kron(np.eye(1), base)
    #     e_0 = self.transition.proj2coord(coord=0)
    #     np.testing.assert_allclose(e_0, e_0_expected)
    #
    #     base = np.zeros(self.transition.num_derivatives + 1)
    #     base[-1] = 1
    #     e_q_expected = np.kron(np.eye(1), base)
    #     e_q = self.transition.proj2coord(coord=self.transition.num_derivatives)
    #     np.testing.assert_allclose(e_q, e_q_expected)

    def test_precon(self):

        assert isinstance(
            self.transition.precon,
            randprocs.markov.integrator.NordsieckLikeCoordinates,
        )

    def test_state_ordering(self):
        assert self.transition.state_ordering in ["coordinate", "derivative"]


def test_only_coordinate_ordering_valid():

    # For the moment, we can only do coordinate-ordering
    with pytest.raises(ValueError):
        randprocs.markov.integrator.IntegratorMixIn(
            num_derivatives=2, state_ordering="derivative"
        )


# Test for the extraction method in the IntegratorMixIn
def test_select_derivative_from_coordinate_ordering():
    """Extract the :math:`i`th derivative from a state."""

    transition = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition(
        driftspeed=0.1, num_derivatives=1, wiener_process_dimension=3
    )
    state_in_coordinate_ordering = np.array(["y1", "dy1", "y2", "dy2", "y3", "dy3"])

    # sanity check for different orderings, refactor this test.
    assert transition.state_ordering == "coordinate"

    received1 = transition.select_derivative(
        state=state_in_coordinate_ordering, derivative=0
    )
    expected1 = np.array(["y1", "y2", "y3"])

    for r, e in zip(received1, expected1):
        assert r == e

    received2 = transition.select_derivative(
        state=state_in_coordinate_ordering, derivative=1
    )
    expected2 = ["dy1", "dy2", "dy3"]
    for r, e in zip(received2, expected2):
        assert r == e


def test_derivative_selection_operator_from_coordinate_ordering():
    """Extract the :math:`i`th derivative from a state."""

    transition = randprocs.markov.integrator.IntegratedOrnsteinUhlenbeckTransition(
        driftspeed=0.1, num_derivatives=1, wiener_process_dimension=3
    )
    state_in_coordinate_ordering = np.array(["y1", "dy1", "y2", "dy2", "y3", "dy3"])

    # sanity check for different orderings, refactor this test.
    assert transition.state_ordering == "coordinate"

    received1 = transition.derivative_selection_operator(derivative=0)
    assert isinstance(received1, np.ndarray)
    expected1 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    np.testing.assert_allclose(received1, expected1)

    received2 = transition.derivative_selection_operator(derivative=1)
    assert isinstance(received2, np.ndarray)
    expected2 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(received2, expected2)
