"""Test cases for the rational quadratic kernel."""

import pytest

from probnum import kernels


@pytest.mark.parametrize("alpha", [-1, -1.0, 0.0, 0])
def test_nonpositive_alpha_raises_exception(alpha):
    """Check whether a non-positive alpha parameter raises a ValueError."""
    with pytest.raises(ValueError):
        kernels.RatQuad(input_dim=1, alpha=alpha)
