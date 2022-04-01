"""Test cases defining random variables with a normal distribution."""

from pytest_cases import case, parametrize

from probnum import backend, linops, randvars
from probnum.backend.typing import ScalarLike, ShapeType
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.typing import MatrixType
import tests.utils


@case(tags=["scalar"])
@parametrize("mean", (0.0, -1.0, 4))
@parametrize("var", (3.0, 2))
def case_scalar(mean: ScalarLike, var: ScalarLike) -> randvars.Normal:
    return randvars.Normal(mean, var)


@case(tags=["vector"])
@parametrize("shape", [(1,), (2,), (5,), (10,)])
def case_vector(shape: ShapeType) -> randvars.Normal:
    seed_mean, seed_cov = backend.random.split(
        tests.utils.random.seed_from_sampling_args(
            base_seed=654,
            shape=shape,
        ),
        num=2,
    )

    return randvars.Normal(
        mean=5.0 * backend.random.standard_normal(seed_mean, shape=shape),
        cov=random_spd_matrix(seed_cov, shape[0]),
    )


@case(tags=["vector", "diag-cov"])
@parametrize(
    "cov", [backend.eye(7, dtype=backend.single), linops.Scaling(2.7, shape=(20, 20))]
)
def case_vector_diag_cov(cov: MatrixType) -> randvars.Normal:
    seed = tests.utils.random.seed_from_sampling_args(
        base_seed=12390,
        shape=cov.shape,
        dtype=cov.dtype,
    )

    return randvars.Normal(
        mean=3.1 * backend.random.standard_normal(seed, shape=cov.shape[0]),
        cov=cov,
    )


@case(tags=["matrix"])
@parametrize("shape", [(1, 1), (5, 1), (1, 4), (2, 2), (3, 4)])
def case_matrix(shape: ShapeType) -> randvars.Normal:
    seed_mean, seed_cov = backend.random.split(
        tests.utils.random.seed_from_sampling_args(
            base_seed=453987,
            shape=shape,
        ),
        num=2,
    )

    return randvars.Normal(
        mean=4.0 * backend.random.standard_normal(seed_mean, shape=shape),
        cov=random_spd_matrix(seed_cov, shape[0] * shape[1]),
    )


@case(tags=["matrix", "mean-op", "cov-op"])
@parametrize("shape", [(1, 1), (2, 1), (1, 3), (2, 2)])
def case_matrix_mean_op_kronecker_cov(shape: ShapeType) -> randvars.Normal:
    seed_mean, seed_cov_A, seed_cov_B = backend.random.split(
        tests.utils.random.seed_from_sampling_args(
            base_seed=421376,
            shape=shape,
        ),
        num=3,
    )

    cov = linops.Kronecker(
        A=random_spd_matrix(seed_cov_A, shape[0]),
        B=random_spd_matrix(seed_cov_B, shape[1]),
    )
    cov.is_symmetric = True
    cov.A.is_symmetric = True
    cov.B.is_symmetric = True

    return randvars.Normal(
        mean=linops.aslinop(backend.random.standard_normal(seed_mean, shape=shape)),
        cov=cov,
    )
