"""
Benchmarks for distributions.
"""

import numpy as np

import probnum.prob as prob
import probnum.linalg.linops as linops

# Module level variables
DISTRIBUTION_NAMES = [
    "univar_normal",
    "multivar_normal",
    "matrixvar_normal",
    "symmatrixvar_normal",
    # "operatorvar_normal"
]


def get_randvar(distribution_name):
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

    # Random variable for a given distribution
    if distribution_name == "univar_normal":
        distribution = prob.Normal(mean=mean_0d, cov=cov_0d)
    elif distribution_name == "multivar_normal":
        distribution = prob.Normal(mean=mean_1d, cov=cov_1d)
    elif distribution_name == "matrixvar_normal":
        distribution = prob.Normal(mean=mean_2d_mat, cov=cov_2d_kron)
    elif distribution_name == "symmatrixvar_normal":
        distribution = prob.Normal(mean=mean_2d_mat, cov=cov_2d_symkron)
    elif distribution_name == "operatorvar_normal":
        distribution = prob.Normal(mean=mean_2d_linop)

    return prob.RandomVariable(distribution=distribution)


class Functions:
    """
    Benchmark various functions of distributions.
    """

    param_names = ["dist", "method"]
    params = [DISTRIBUTION_NAMES, ["pdf", "logpdf", "cdf", "logcdf"]]

    def setup(self, dist, method):
        # pylint: disable=missing-function-docstring,attribute-defined-outside-init
        # pylint: disable=unused-argument
        self.randvar = get_randvar(distribution_name=dist)
        self.eval_point = np.random.uniform(size=self.randvar.shape)
        self.quantile = np.random.uniform(size=self.randvar.shape)

    def time_distr_functions(self, dist, method):
        """Time evaluation of the pdf, logpdf, cdf and logcdf."""
        # pylint: disable=unused-argument
        try:
            if method == "pdf":
                self.randvar.distribution.pdf(x=self.eval_point)
            elif method == "logpdf":
                self.randvar.distribution.logpdf(x=self.eval_point)
            elif method == "cdf":
                self.randvar.distribution.cdf(x=self.quantile)
            elif method == "logcdf":
                self.randvar.distribution.logcdf(x=self.quantile)
        except NotImplementedError:
            pass


class Sampling:
    """
    Benchmark sampling routines for various distributions.
    """

    param_names = ["dist"]
    params = [DISTRIBUTION_NAMES]

    def setup(self, dist):
        # pylint: disable=missing-function-docstring,attribute-defined-outside-init
        np.random.seed(42)
        self.n_samples = 1000
        self.randvar = get_randvar(distribution_name=dist)

    def time_sample(self, dist):
        """Time sampling from this distribution."""
        # pylint: disable=unused-argument
        self.randvar.sample(self.n_samples)

    def peakmem_sample(self, dist):
        """Peak memory of sampling process."""
        # pylint: disable=unused-argument
        self.randvar.sample(self.n_samples)
