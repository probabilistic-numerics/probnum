import numpy as np
from scipy.optimize._numdiff import approx_derivative

from probnum import backend, compat, functions, randprocs, randvars

import pytest


def assert_gradient_approx_finite_differences(
    func,
    grad,
    x0,
    *,
    epsilon=None,
    method="3-point",
    rtol=1e-7,
    atol=0.0,
):
    if epsilon is None:
        out = func(x0)

        epsilon = backend.sqrt(backend.finfo(out.dtype).eps)

    compat.testing.assert_allclose(
        grad(x0),
        approx_derivative(
            lambda x: backend.asarray(func(x), copy=False),
            x0,
            method=method,
        ),
        rtol=rtol,
        atol=atol,
    )


def g(l):
    l = l[0]

    gp = randprocs.GaussianProcess(
        mean=functions.Zero(input_shape=()),
        cov=randprocs.kernels.ExpQuad(input_shape=(), lengthscale=l),
    )

    xs = backend.linspace(-1.0, 1.0, 10)
    ys = backend.linspace(-1.0, 1.0, 10)

    fX = gp(xs)

    e = randvars.Normal(mean=backend.zeros(10), cov=backend.eye(10))

    return -(fX + e).logpdf(ys)


@pytest.mark.skipif_backend(backend.Backend.NUMPY)
def test_compare_grad():
    l = backend.asarray([3.0])
    dg = backend.autodiff.grad(g)

    assert_gradient_approx_finite_differences(
        g,
        dg,
        x0=l,
    )


if __name__ == "__main__":
    test_compare_grad()
