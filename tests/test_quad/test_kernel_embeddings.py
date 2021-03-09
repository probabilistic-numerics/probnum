"""Test cases for kernel embeddings."""

import numpy as np
from scipy.integrate import dblquad, quad

import probnum.quad._integration_measures as measures


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
    """Test kernel mean against scipy.integrate.quad or sampling."""
    if kernel_embedding.dim != 1:
        # use MC sampling for ground truth
        n_mc = 10000
        X = kernel_embedding.measure.sample(n_mc)
        num_kmeans = kernel_embedding.kernel(x.T, X).sum(axis=1) / n_mc
        true_kmeans = kernel_embedding.kernel_mean(x)

        np.testing.assert_allclose(true_kmeans, num_kmeans, rtol=1.0e-3, atol=1.0e-3)

    else:
        # numerical solution
        num_kmeans = np.empty((x.shape[1],))

        # get reasonable integration box to avoid trouble with quad and np.inf
        mean = kernel_embedding.measure.mean
        std = np.sqrt(kernel_embedding.measure.cov)
        integration_box = (mean - 5 * std, mean + 5 * std)

        for i in range(x.shape[1]):

            def f(xx):
                return kernel_embedding.kernel(
                    np.float(x[:, i]), xx
                ) * kernel_embedding.measure(xx)

            num_kmeans[i] = quad(f, integration_box[0], integration_box[1])[0]

        # closed form solution
        true_kmeans = kernel_embedding.kernel_mean(x)

        np.testing.assert_allclose(true_kmeans, num_kmeans, rtol=1.0e-6, atol=1.0e-6)


def test_kvar(kernel_embedding):
    """Test kernel mean against scipy.integrate.dblquad or sampling."""
    if kernel_embedding.dim != 1:
        n_mc = 10000
        X = kernel_embedding.measure.sample(n_mc)
        num_kvar = kernel_embedding.kernel_mean(X.T).sum() / n_mc
        true_kvar = kernel_embedding.kernel_variance()

        np.testing.assert_allclose(true_kvar, num_kvar, rtol=1.0e-3, atol=1.0e-3)
    else:
        # numerical solution
        def f(x, y):
            return (
                kernel_embedding.kernel(x, y)
                * kernel_embedding.measure(x)
                * kernel_embedding.measure(y)
            )

        num_integral = dblquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[
            0
        ]
        true_integral = kernel_embedding.kernel_variance()

        np.testing.assert_approx_equal(
            num_integral,
            true_integral,
            significant=5,
            err_msg="closed form and numerical integral are off",
            verbose=False,
        )
