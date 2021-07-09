"""Benchmarks for random variables."""

import numpy as np

from probnum import linops
from probnum import randvars as rvs

# Module level variables
RV_NAMES = [
    "univar_normal",
    "multivar_normal",
    "matrixvar_normal",
    "symmatrixvar_normal",
    "operatorvar_normal",
]


SPD_MATRIX_5x5 = np.array(
    [
        [2.3, -2.3, 3.5, 4.2, 1.8],
        [-2.3, 3.0, -3.5, -4.8, -1.9],
        [3.5, -3.5, 6.9, 5.8, 0.8],
        [4.2, -4.8, 5.8, 10.1, 6.3],
        [1.8, -1.9, 0.8, 6.3, 12.1],
    ]
)


def get_randvar(rv_name):
    """Return a random variable for a given distribution name."""
    # Distribution Means and Covariances

    mean_0d = np.random.rand()
    mean_1d = np.random.rand(5)
    mean_2d_mat = SPD_MATRIX_5x5
    mean_2d_linop = linops.Matrix(SPD_MATRIX_5x5)
    cov_0d = np.random.rand() + 10 ** -12
    cov_1d = SPD_MATRIX_5x5
    cov_2d_kron = linops.Kronecker(A=SPD_MATRIX_5x5, B=SPD_MATRIX_5x5)
    cov_2d_symkron = linops.SymmetricKronecker(A=SPD_MATRIX_5x5)

    if rv_name == "univar_normal":
        randvar = rvs.Normal(mean=mean_0d, cov=cov_0d)
    elif rv_name == "multivar_normal":
        randvar = rvs.Normal(mean=mean_1d, cov=cov_1d)
    elif rv_name == "matrixvar_normal":
        randvar = rvs.Normal(mean=mean_2d_mat, cov=cov_2d_kron)
    elif rv_name == "symmatrixvar_normal":
        randvar = rvs.Normal(mean=mean_2d_mat, cov=cov_2d_symkron)
    elif rv_name == "operatorvar_normal":
        randvar = rvs.Normal(mean=mean_2d_linop, cov=cov_2d_symkron)
    else:
        raise ValueError("Random variable not found.")

    return randvar


class Functions:
    """Benchmark various functions of random variables."""

    param_names = ["randvar", "method"]
    params = [RV_NAMES, ["pdf", "logpdf", "cdf", "logcdf"]]

    def setup(self, randvar, method):
        self.randvar = get_randvar(rv_name=randvar)
        self.eval_point = np.random.uniform(size=self.randvar.shape)
        self.quantile = np.random.uniform(size=self.randvar.shape)

    def time_distr_functions(self, randvar, method):
        """Times evaluation of the pdf, logpdf, cdf and logcdf."""
        try:
            if method == "pdf":
                self.randvar.pdf(x=self.eval_point)
            elif method == "logpdf":
                self.randvar.logpdf(x=self.eval_point)
            elif method == "cdf":
                self.randvar.cdf(x=self.quantile)
            elif method == "logcdf":
                self.randvar.logcdf(x=self.quantile)
        except NotImplementedError:
            pass


class Sampling:
    """Benchmark sampling routines for various distributions."""

    param_names = ["randvar"]
    params = [RV_NAMES]

    def setup(self, randvar):
        self.rng = np.random.default_rng(seed=2)
        self.n_samples = 1000
        self.randvar = get_randvar(rv_name=randvar)

    def time_sample(self, randvar):
        """Times sampling from this distribution."""
        self.randvar.sample(rng=self.rng, size=self.n_samples)

    def peakmem_sample(self, randvar):
        """Peak memory of sampling process."""
        self.randvar.sample(rng=self.rng, size=self.n_samples)
