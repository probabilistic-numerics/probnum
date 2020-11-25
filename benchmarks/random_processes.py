"""Benchmarks for random processes."""

import numpy as np

from probnum import kernels as kerns
from probnum import random_processes as rps

# Module level variables
RP_NAMES = ["gaussian"]


def get_randproc(rp_name, input_dim, output_dim):
    """Return a random process for a given name."""

    def mean_zero(x, out_dim=1):
        x = np.asarray(x)
        if x.ndim > 1:
            shape = (x.shape[0], out_dim)
        else:
            shape = (out_dim,)

        return np.zeros(shape=shape)

    if rp_name == "gaussian":
        randproc = rps.GaussianProcess(
            mean=mean_zero,
            cov=kerns.ExpQuad(input_dim=input_dim),
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown random process '{rp_name}'.")

    return randproc


class Functions:
    """Benchmark various functions of random processes."""

    param_names = ["randproc", "input_dim"]
    params = [RP_NAMES, [1, 10, 100]]

    def setup(self, randproc, input_dim):
        self.rng = np.random.default_rng(41)
        self.randproc = get_randproc(
            rp_name=randproc, input_dim=input_dim, output_dim=1
        )
        self.x = self.rng.normal((1000, input_dim))

    def time_eval(self, randproc, input_dim):
        """Time evaluation of the random process at a set of inputs."""
        self.randproc(self.x)

    def time_mean(self, randproc, input_dim):
        self.randproc.mean(self.x)

    def time_cov(self, randproc, input_dim):
        self.randproc.cov(self.x)

    def time_var(self, randproc, input_dim):
        self.randproc.var(self.x)


class Sampling:
    """Benchmark sampling routines for various distributions."""

    param_names = ["randproc", "input_dim", "n_samples"]
    params = [RP_NAMES, [1, 10, 100], [10, 100, 1000]]

    def setup(self, randproc, input_dim, n_samples):
        np.random.seed(42)
        self.n_samples = n_samples
        self.randproc = get_randproc(
            rp_name=randproc, input_dim=input_dim, output_dim=1
        )

    def time_sample(self, randproc, input_dim, n_samples):
        """Times sampling from this distribution."""
        self.randproc.sample(size=self.n_samples)

    def peakmem_sample(self, randproc, input_dim, n_samples):
        """Peak memory of sampling process."""
        self.randproc.sample(size=self.n_samples)
