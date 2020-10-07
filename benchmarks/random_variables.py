"""
Benchmarks for random variables.
"""

import numpy as np

import probnum.linear_operators as linear_operators
from benchmarks.benchmark_utils import SPD_MATRIX_5x5
from probnum import random_variables as rvs

# Module level variables
RV_NAMES = [
    "univar_normal",
    "multivar_normal",
    "matrixvar_normal",
    "symmatrixvar_normal",
    "operatorvar_normal",
]


def get_randvar(rv_name):
    """Return a random variable for a given distribution name."""
    # Distribution Means and Covariances

    mean_0d = np.random.rand()
    mean_1d = np.random.rand(5)
    mean_2d_mat = SPD_MATRIX_5x5
    mean_2d_linop = linear_operators.MatrixMult(SPD_MATRIX_5x5)
    cov_0d = np.random.rand() + 10 ** -12
    cov_1d = SPD_MATRIX_5x5
    cov_2d_kron = linear_operators.Kronecker(A=SPD_MATRIX_5x5, B=SPD_MATRIX_5x5)
    cov_2d_symkron = linear_operators.SymmetricKronecker(A=SPD_MATRIX_5x5)

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

    return randvar


class Functions:
    """Benchmark various functions of random variables."""

    param_names = ["randvar", "method"]
    params = [RV_NAMES, ["pdf", "logpdf", "cdf", "logcdf"]]

    def setup(self, randvar, method):
        # pylint: disable=unused-argument,attribute-defined-outside-init,missing-function-docstring

        self.randvar = get_randvar(rv_name=randvar)
        self.eval_point = np.random.uniform(size=self.randvar.shape)
        self.quantile = np.random.uniform(size=self.randvar.shape)

    def time_distr_functions(self, randvar, method):
        """Times evaluation of the pdf, logpdf, cdf and logcdf."""
        # pylint: disable=unused-argument

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
        # pylint: disable=unused-argument,attribute-defined-outside-init,missing-function-docstring
        np.random.seed(42)
        self.n_samples = 1000
        self.randvar = get_randvar(rv_name=randvar)

    def time_sample(self, randvar):
        """Times sampling from this distribution."""
        # pylint: disable=unused-argument

        self.randvar.sample(self.n_samples)

    def peakmem_sample(self, randvar):
        """Peak memory of sampling process."""
        # pylint: disable=unused-argument

        self.randvar.sample(self.n_samples)
