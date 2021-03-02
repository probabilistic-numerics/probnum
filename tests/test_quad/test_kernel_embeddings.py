"""Test cases for kernel embeddings."""

import numpy as np


def test_kmean_shape(kernel_embedding, x):
    """Test output shape of kernel mean."""

    kmean_shape = (x.shape[1],)

    assert kernel_embedding.kernel_mean(x).shape == kmean_shape, (
        f"Kernel mean of {type(kernel_embedding)} has shape {kernel_embedding.kernel_mean(x).shape} instead of"
        f" {kmean_shape}"
    )


def test_kvar_float(kernel_embedding):
    """Test output of kernel variance."""
    assert isinstance(kernel_embedding.kernel_variance(), np.float)


def test_kmean(kernel_embedding, x):
    """Test kernel mean against scipy.integrate.quad."""
    # TODO: requires evaluation of IntegrationMeasure, which is not yet implemented
    pass


def test_kvar(kernel_embedding):
    """Test kernel mean against scipy.integrate.dblquad."""
    # TODO: requires evaluation of IntegrationMeasure, which is not yet implemented
    pass
