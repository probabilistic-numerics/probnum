"""Test cases for the rational quadratic kernel."""

import pytest

from probnum.randprocs import kernels


@pytest.mark.parametrize("alpha", [-1, -1.0, 0.0, 0])
def test_nonpositive_alpha_raises_exception(alpha: float):
    """Check whether a non-positive alpha parameter raises a ValueError."""
    with pytest.raises(ValueError):
        kernels.RatQuad(input_shape=(), alpha=alpha)
