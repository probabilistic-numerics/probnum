"""Basic tests for automatic differentiation functionality."""
from probnum import backend, compat
from probnum.backend.autodiff import grad, hessian, jacfwd, jacrev

import pytest


@pytest.mark.skipif_backend(backend.Backend.NUMPY)
@pytest.mark.parametrize("x", backend.linspace(0, 2 * backend.pi, 10))
def test_grad_basic_function(x: backend.Array):
    compat.testing.assert_allclose(grad(backend.sin)(x), backend.cos(x))


@pytest.mark.skipif_backend(backend.Backend.NUMPY)
@pytest.mark.parametrize("x", backend.linspace(0, 2 * backend.pi, 10))
def test_jacfwd_basic_function(x: backend.Array):
    compat.testing.assert_allclose(jacfwd(backend.sin)(x), backend.cos(x))


@pytest.mark.skipif_backend(backend.Backend.NUMPY)
@pytest.mark.parametrize("x", backend.linspace(0, 2 * backend.pi, 10))
def test_jacrev_basic_function(x: backend.Array):
    compat.testing.assert_allclose(jacrev(backend.sin)(x), backend.cos(x))


@pytest.mark.skipif_backend(backend.Backend.NUMPY)
@pytest.mark.parametrize("x", backend.linspace(0, 2 * backend.pi, 10))
def test_hessian_basic_function(x: backend.Array):
    compat.testing.assert_allclose(hessian(backend.sin)(x), -backend.sin(x))
