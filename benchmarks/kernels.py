"""Benchmarks for random variables."""

import numpy as np

import probnum.kernels as kernels

# Module level variables
KERNEL_NAMES = [
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


def get_kernel(kernel_name, input_dim):
    """Return a kernel for a given name."""
    if kernel_name == "white_noise":
        kernel = kernels.WhiteNoise(input_dim=input_dim)
    elif kernel_name == "linear":
        kernel = kernels.Linear(input_dim=input_dim)
    elif kernel_name == "polynomial":
        kernel = kernels.Polynomial(input_dim=input_dim)
    elif kernel_name == "exp_quad":
        kernel = kernels.ExpQuad(input_dim=input_dim)
    elif kernel_name == "rat_quad":
        kernel = kernels.RatQuad(input_dim=input_dim)
    elif kernel_name == "matern12":
        kernel = kernels.Matern(input_dim=input_dim, nu=0.5)
    elif kernel_name == "matern32":
        kernel = kernels.Matern(input_dim=input_dim, nu=1.5)
    elif kernel_name == "matern52":
        kernel = kernels.Matern(input_dim=input_dim, nu=2.5)
    elif kernel_name == "matern72":
        kernel = kernels.Matern(input_dim=input_dim, nu=3.5)
    else:
        raise ValueError(f"Kernel name '{kernel_name}' not recognized.")

    return kernel


class Kernels:
    """Benchmark evaluation of a kernel at a set of inputs."""

    param_names = ["kernel", "n_datapoints"]
    params = [KERNEL_NAMES, N_DATAPOINTS]

    def setup(self, kernel, n_datapoints):
        rng = np.random.default_rng(42)
        self.input_dim = 100
        self.data = rng.normal(size=(n_datapoints, self.input_dim))
        self.kernel = get_kernel(kernel_name=kernel, input_dim=self.input_dim)

    def time_kernel_matrix(self, kernel, n_datapoints):
        """Times sampling from this distribution."""
        self.kernel(self.data)

    def peakmem_kernel_matrix(self, kernel, n_datapoints):
        """Peak memory of sampling process."""
        self.kernel(self.data)
