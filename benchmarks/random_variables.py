"""
Benchmarks for random variables.
"""

import numpy as np

from probnum.prob import random_variable as rvars
import probnum.linalg.linops as linops

# Module level variables
rv_names = [
    "univar_normal",
    "multivar_normal",
    "matrixvar_normal",
    "symmatrixvar_normal",
    # "operatorvar_normal"
]


def get_randvar(rv_name):
    """
    Return a random variable for a given distribution name
    """
    # Distribution Means and Covariances
    spd_mat = np.array(
        [
            [2.3, -2.3, 3.5, 4.2, 1.8],
            [-2.3, 3.0, -3.5, -4.8, -1.9],
            [3.5, -3.5, 6.9, 5.8, 0.8],
            [4.2, -4.8, 5.8, 10.1, 6.3],
            [1.8, -1.9, 0.8, 6.3, 12.1],
        ]
    )
    mean_0d = np.random.rand()
    mean_1d = np.random.rand(5)
    mean_2d_mat = spd_mat
    mean_2d_linop = linops.MatrixMult(spd_mat)
    cov_0d = np.random.rand() + 10 ** -12
    cov_1d = spd_mat
    cov_2d_kron = linops.Kronecker(A=spd_mat, B=spd_mat)
    cov_2d_symkron = linops.SymmetricKronecker(A=spd_mat)

    if rv_name == "univar_normal":
        randvar = rvars.Normal(mean=mean_0d, cov=cov_0d)
    elif rv_name == "multivar_normal":
        randvar = rvars.Normal(mean=mean_1d, cov=cov_1d)
    elif rv_name == "matrixvar_normal":
        randvar = rvars.Normal(mean=mean_2d_mat, cov=cov_2d_kron)
    elif rv_name == "symmatrixvar_normal":
        randvar = rvars.Normal(mean=mean_2d_mat, cov=cov_2d_symkron)
    elif rv_name == "operatorvar_normal":
        randvar = rvars.Normal(mean=mean_2d_linop, cov=cov_2d_symkron)

    return randvar


class Functions:
    """
    Benchmark various functions of distributions.
    """

    param_names = ["rv_name", "property"]
    params = [rv_names, ["pdf", "logpdf", "cdf", "logcdf"]]

    def setup(self, rv_name, property):
        self.randvar = get_randvar(rv_name=rv_name)
        self.eval_point = np.random.uniform(self.randvar.shape)
        self.quantile = np.random.uniform(self.randvar.shape)

    def time_distr_functions(self, rv_name, property):
        """Times evaluation of the pdf, logpdf, cdf and logcdf."""
        try:
            if property == "pdf":
                self.randvar.pdf(x=self.eval_point)
            elif property == "logpdf":
                self.randvar.logpdf(x=self.eval_point)
            elif property == "cdf":
                self.randvar.cdf(x=self.quantile)
            elif property == "logcdf":
                self.randvar.logcdf(x=self.quantile)
        except NotImplementedError:
            pass


class Sampling:
    """
    Benchmark sampling routines for various distributions.
    """

    param_names = ["rv_name"]
    params = [rv_names]

    def setup(self, rv_name):
        np.random.seed(42)
        self.n_samples = 1000
        self.randvar = get_randvar(rv_name=rv_name)

    def time_sample(self, rv_name):
        """Times sampling from this distribution."""
        self.randvar.sample(self.n_samples)

    def peakmem_sample(self, rv_name):
        """Peak memory of sampling process."""
        self.randvar.sample(self.n_samples)
