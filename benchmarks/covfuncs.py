"""Benchmarks for covariance functions."""

import numpy as np

from probnum.randprocs import covfuncs

# Module level variables
COVFUNC_NAMES = [
    "white_noise",
    "linear",
    "polynomial",
    "exp_quad",
    "rat_quad",
    "matern12",
    "matern32",
    "matern52",
    "matern72",
]

N_DATAPOINTS = [10, 100, 1000]


def get_covfunc(covfunc_name, input_shape):
    """Return a covariance function for a given name."""
    if covfunc_name == "white_noise":
        k = covfuncs.WhiteNoise(input_shape=input_shape)
    elif covfunc_name == "linear":
        k = covfuncs.Linear(input_shape=input_shape)
    elif covfunc_name == "polynomial":
        k = covfuncs.Polynomial(input_shape=input_shape)
    elif covfunc_name == "exp_quad":
        k = covfuncs.ExpQuad(input_shape=input_shape)
    elif covfunc_name == "rat_quad":
        k = covfuncs.RatQuad(input_shape=input_shape)
    elif covfunc_name == "matern12":
        k = covfuncs.Matern(input_shape=input_shape, nu=0.5)
    elif covfunc_name == "matern32":
        k = covfuncs.Matern(input_shape=input_shape, nu=1.5)
    elif covfunc_name == "matern52":
        k = covfuncs.Matern(input_shape=input_shape, nu=2.5)
    elif covfunc_name == "matern72":
        k = covfuncs.Matern(input_shape=input_shape, nu=3.5)
    else:
        raise ValueError(f"Covariance function '{covfunc_name}' not recognized.")

    return k


class CovarianceFunctions:
    """Benchmark evaluation of a covariance function at a set of inputs."""

    param_names = ["covfunc", "n_datapoints"]
    params = [COVFUNC_NAMES, N_DATAPOINTS]

    def setup(self, covfunc, n_datapoints):
        rng = np.random.default_rng(42)
        self.input_dim = 100
        self.data = rng.normal(size=(n_datapoints, self.input_dim))
        self.covfunc = get_covfunc(covfunc_name=covfunc, input_shape=self.input_dim)

    def time_covfunc_call(self, covfunc, n_datapoints):
        self.covfunc(self.data, None)

    def time_covfunc_matrix(self, covfunc, n_datapoints):
        """Times sampling from this distribution."""
        self.covfunc.matrix(self.data)

    def peakmem_covfunc_matrix(self, covfunc, n_datapoints):
        """Peak memory of sampling process."""
        self.covfunc.matrix(self.data)
