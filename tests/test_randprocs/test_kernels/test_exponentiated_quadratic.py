"""Test cases for the `ExpQuad` kernel."""

import numpy as np
import pytest

from probnum.randprocs import kernels


@pytest.mark.parametrize("lengthscales", (0.0, -1.0, (0.0, 1.0), (-0.2, 2.0)))
def test_nonpositive_lengthscales_raises_exception(lengthscales):
    """Check whether a non-positive `lengthscales` parameter raises a ValueError."""
    with pytest.raises(ValueError):
        kernels.ExpQuad(np.shape(lengthscales), lengthscales=lengthscales)
