import numpy as np
from scipy.optimize._numdiff import approx_derivative

import probnum as pn
from probnum import backend


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

        epsilon = np.sqrt(backend.finfo(out.dtype).eps)

    np.testing.assert_allclose(
        np.array(grad(x0)),
        approx_derivative(
            lambda x: np.array(func(x), copy=False),
            x0,
            method=method,
        ),
        rtol=rtol,
        atol=atol,
    )


def g(l):
    l = l[0]

    gp = pn.randprocs.GaussianProcess(
        mean=lambda x: backend.zeros_like(x, shape=x.shape[:-1]),
        cov=pn.kernels.ExpQuad(input_dim=1, lengthscale=l),
    )

    xs = backend.linspace(-1.0, 1.0, 10)
    ys = backend.linspace(-1.0, 1.0, 10)

    fX = gp(xs[:, None])

    e = pn.randvars.Normal(mean=backend.zeros(10), cov=backend.eye(10))

    return -(fX + e).logpdf(ys)


def test_compare_grad():
    l = backend.ones((1,)) * 3.0
    dg = backend.autodiff.grad(g)

    assert_gradient_approx_finite_differences(
        g,
        dg,
        x0=l,
    )


if __name__ == "__main__":
    test_compare_grad()