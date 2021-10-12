"""Benchmarks for linear solvers."""
import numpy as np

from probnum import linops, problems, randvars
from probnum.linalg import problinsolve
from probnum.problems.zoo.linalg import random_sparse_spd_matrix, random_spd_matrix

LINEAR_SYSTEMS = ["dense", "sparse", "linop"]
LINSYS_DIMS = [100, 1000, 10000, 100000]
QUANTITIES_OF_INTEREST = ["x", "A", "Ainv"]


def get_linear_system(name: str, dim: int):
    rng = np.random.default_rng(0)

    if name == "dense":
        if dim > 1000:
            raise NotImplementedError()
        A = random_spd_matrix(rng=rng, dim=dim)
    elif name == "sparse":
        A = random_sparse_spd_matrix(
            rng=rng, dim=dim, density=np.minimum(1.0, 1000 / dim ** 2)
        )
    elif name == "linop":
        if dim > 100:
            raise NotImplementedError()
            # TODO: Larger benchmarks currently fail. Remove once PLS refactor (https://github.com/probabilistic-numerics/probnum/issues/51) is resolved
        A = linops.Scaling(factors=rng.normal(size=(dim,)))
    else:
        raise NotImplementedError()

    solution = rng.normal(size=(dim,))
    b = A @ solution
    return problems.LinearSystem(A=A, b=b, solution=solution)


def get_quantity_of_interest(
    qoi: str,
    x: randvars.RandomVariable,
    A: randvars.RandomVariable,
    Ainv: randvars.RandomVariable,
):
    if qoi == "x":
        return x
    elif qoi == "A":
        return A
    elif qoi == "Ainv":
        return Ainv
    else:
        raise NotImplementedError()


class LinSolve:
    """Benchmark solving a linear system."""

    param_names = ["linsys", "dim"]
    params = [
        LINEAR_SYSTEMS,
        LINSYS_DIMS,
    ]

    def setup(self, linsys, dim):
        self.linsys = get_linear_system(name=linsys, dim=dim)
        xhat, _, _, _ = problinsolve(A=self.linsys.A, b=self.linsys.b)
        self.xhat = xhat

    def time_solve(self, linsys, dim):
        problinsolve(A=self.linsys.A, b=self.linsys.b)

    def peakmem_solve(self, linsys, dim):
        problinsolve(A=self.linsys.A, b=self.linsys.b)

    def track_residual_norm(self, linsys, dim):
        return np.linalg.norm(self.linsys.A @ self.xhat.mean - self.linsys.b)

    def track_error_2norm(self, linsys, dim):
        return np.linalg.norm(self.linsys.solution - self.xhat.mean)

    def track_error_Anorm(self, linsys, dim):
        diff = self.linsys.solution - self.xhat.mean
        return np.sqrt(np.inner(diff, self.linsys.A @ diff))


class PosteriorBelief:
    """Benchmark computing derived quantities from the posterior belief."""

    param_names = ["linsys", "dim", "qoi"]
    params = [LINEAR_SYSTEMS, LINSYS_DIMS, QUANTITIES_OF_INTEREST]

    def setup(self, linsys, dim, qoi):

        if dim > 1000:
            # Operations on the posterior for large matrices can be very memory intensive.
            raise NotImplementedError()

        self.linsys = get_linear_system(name=linsys, dim=dim)
        x, A, Ainv, _ = problinsolve(A=self.linsys.A, b=self.linsys.b)
        self.qoi = get_quantity_of_interest(qoi, x, A, Ainv)

    def time_trace_cov(self, linsys, dim, qoi):
        self.qoi.cov.trace()

    def peakmem_trace_cov(self, linsys, dim, qoi):
        self.qoi.cov.trace()
