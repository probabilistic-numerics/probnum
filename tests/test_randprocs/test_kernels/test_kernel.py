"""Generic tests for kernels."""

import copy
import numpy as np
import pytest

from probnum.randprocs import kernels


@pytest.mark.parametrize("sigma_sq", [-2.0, -0.5, 0.0])
def test_nonpositive_sigma_sq_raises_value_error(sigma_sq, kernel: kernels.Kernel):
    with pytest.raises(ValueError):
        kernel.sigma_sq = sigma_sq


@pytest.mark.parametrize("sigma_sq", [0.4, 1.0, 2.1])
def test_scaling(
    sigma_sq,
    kernel: kernels.Kernel,
    x0: np.ndarray,
):
    """Check multiplication by the squared scaling parameter is equivalent to
    evaluating the scaled kernel."""
    kernel_unscaled = copy.copy(kernel)
    kernel_unscaled.sigma_sq = 1.0
    kernel.sigma_sq = sigma_sq
    assert np.allclose(
        kernel(x0, x0), sigma_sq * kernel_unscaled(x0, x0), rtol=1e-16, atol=1e-16
    )
