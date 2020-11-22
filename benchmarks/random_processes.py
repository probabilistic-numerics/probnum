"""Benchmarks for random processes."""

import numpy as np

from probnum import random_processes as rps

# Module level variables
RP_NAMES = ["gaussian"]


def get_randproc(rp_name):
    """Return a random process for a given name."""

    if rp_name == "gaussian":
        randproc = rps.GaussianProcess()

    return randproc


class Functions:
    """Benchmark various functions of random processes."""

    param_names = ["randvar", "method"]
    params = [RP_NAMES]

    def setup(self, randproc, method):
        # pylint: disable=unused-argument,missing-function-docstring

        self.randvar = get_randproc(rp_name=randproc)

    def time_eval(self):
        """Time evaluation of the random process at a set of inputs."""
        pass

    def time_mean(self):
        pass

    def time_cov(self):
        pass


class Sampling:
    """Benchmark sampling routines for various distributions."""

    param_names = ["randproc"]
    params = [RP_NAMES]

    def setup(self, randvar):
        # pylint: disable=unused-argument,attribute-defined-outside-init,missing-function-docstring
        np.random.seed(42)
        self.n_samples = 1000
        self.randproc = get_randproc(rp_name=randvar)

    def time_sample(self, randproc):
        """Times sampling from this distribution."""
        # pylint: disable=unused-argument

        self.randproc.sample(self.n_samples)

    def peakmem_sample(self, randproc):
        """Peak memory of sampling process."""
        # pylint: disable=unused-argument

        self.randproc.sample(self.n_samples)
