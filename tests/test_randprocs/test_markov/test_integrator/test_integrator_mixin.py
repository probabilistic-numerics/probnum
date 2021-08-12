import numpy as np
import pytest

from probnum import randprocs


class IntegratorMixInTestMixIn:
    def test_uses_mixin(self):
        assert isinstance(self.transition, randprocs.markov.integrator.IntegratorMixIn)

    def test_num_derivatives(self):

        # Check for > 0, because we do not know how many the test cases generate.
        # If this property does not exist, however, this test will fail.
        assert self.transition.num_derivatives > 0

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

    # Dummy transition that uses the Mixin
    transition = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=1, wiener_process_dimension=3
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

    # Dummy transition that uses the Mixin
    transition = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=1, wiener_process_dimension=3
    )

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


def test_reorder_states():
    arr = np.array(["y1", "y2", "y3", "dy1", "dy2", "dy3"])

    # Dummy transition that uses the IntegratorMixIn
    transition = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=1, wiener_process_dimension=3
    )

    new_arr = transition.reorder_state(
        arr, current_ordering="derivative", target_ordering="coordinate"
    )
    expected = np.array(["y1", "dy1", "y2", "dy2", "y3", "dy3"])
    for r, e in zip(new_arr, expected):
        assert r == e

    old_arr = transition.reorder_state(
        new_arr, current_ordering="coordinate", target_ordering="derivative"
    )
    for r, e in zip(old_arr, arr):
        assert r == e

    with pytest.raises(ValueError):
        transition.reorder_state(
            arr, current_ordering="derivative", target_ordering="something"
        )


def test_reorder_states_identity():
    arr = np.array(["y1", "y2", "y3", "dy1", "dy2", "dy3"])

    # Dummy transition that uses the IntegratorMixIn
    transition = randprocs.markov.integrator.IntegratedWienerTransition(
        num_derivatives=1, wiener_process_dimension=3
    )

    for ordering in ["derivative", "coordinate"]:
        new_arr = transition.reorder_state(
            arr, current_ordering=ordering, target_ordering=ordering
        )
        for r, e in zip(new_arr, arr):
            assert r == e
