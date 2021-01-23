"""Tests for the uncertainty scales of the unexplored null spaces of a probabilistic
linear solver."""

import pytest

from probnum.linalg.solvers.hyperparams import UncertaintyUnexploredSpace


@pytest.mark.parametrize("Phi,Psi", [(-1.0, 1.0), (0.5, -0.0001)])
def test_negative_scales_raise_value_error(Phi: float, Psi: float):
    """Test whether negative uncertainty scales raise a ValueError."""
    with pytest.raises(ValueError):
        UncertaintyUnexploredSpace(Phi=Phi, Psi=Psi)
