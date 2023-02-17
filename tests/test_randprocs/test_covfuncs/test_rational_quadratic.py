"""Test cases for the rational quadratic covariance function."""

import pytest

from probnum.randprocs import covfuncs


@pytest.mark.parametrize("alpha", [-1, -1.0, 0.0, 0])
def test_nonpositive_alpha_raises_exception(alpha):
    """Check whether a non-positive alpha parameter raises a ValueError."""
    with pytest.raises(ValueError):
        covfuncs.RatQuad(input_shape=(), alpha=alpha)
