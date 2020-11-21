"""Benchmarks for random variables."""

import numpy as np

import probnum.kernels as kernels

# Module level variables
KERNEL_NAMES = ["white_noise", "linear", "polynomial", "exp_quad", "rat_quad"]


def get_kernel(kernel_name):
    """Return a kernel for a given name."""
    if kernel_name == "white_noise":
        kernel = kernels.WhiteNoise()
    elif kernel_name == "linear":
        kernel = kernels.Linear()
    elif kernel_name == "polynomial":
        kernel = kernels.Polynomial()
    elif kernel_name == "exp_quad":
        kernel = kernels.ExpQuad()
    elif kernel_name == "rat_quad":
        kernel = kernels.RatQuad()
    else:
        raise ValueError(f"Kernel name '{kernel_name}' not recognized.")

    return kernel


class Kernels:
    """Benchmark evaluation of a kernel at a set of inputs."""

    param_names = ["kernel"]
    params = [KERNEL_NAMES]

    def setup(self, kernel):
        # pylint: missing-function-docstring
        rng = np.random.default_rng(42)
        self.data = rng.normal(size=(1000, 100))
        self.kernel = get_kernel(kernel_name=kernel)

    def time_kernel_matrix(self, kernel):
        """Times sampling from this distribution."""
        # pylint: disable=unused-argument
        self.kernel(self.data)

    def peakmem_kernel_matrix(self, kernel):
        """Peak memory of sampling process."""
        # pylint: disable=unused-argument
        self.kernel(self.data)
