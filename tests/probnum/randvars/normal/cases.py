"""Test cases defining random variables with a normal distribution."""

from probnum import backend, linops, randvars
from probnum.backend.typing import ScalarLike, ShapeType
from probnum.problems.zoo.linalg import random_spd_matrix
from probnum.typing import MatrixType

from pytest_cases import case, parametrize
import tests.utils


@case(tags=["scalar"])
@parametrize(mean=[0.0, -1.0, 4])
@parametrize(var=[3.0, 2])
def case_scalar(mean: ScalarLike, var: ScalarLike) -> randvars.Normal:
    return randvars.Normal(mean, var)


@case(tags=["scalar", "degenerate", "constant"])
@parametrize(mean=[0.0, 12.23])
def case_scalar_constant(mean: ScalarLike) -> randvars.Normal:
    return randvars.Normal(mean=mean, cov=0.0)


@case(tags=["vector"])
@parametrize(shape=[(1,), (2,), (5,), (10,)])
def case_vector(shape: ShapeType) -> randvars.Normal:
    rng_state_mean, rng_state_cov = backend.random.split(
        tests.utils.random.rng_state_from_sampling_args(
            base_seed=654,
            shape=shape,
        ),
        num=2,
    )

    return randvars.Normal(
        mean=5.0 * backend.random.standard_normal(rng_state_mean, shape=shape),
        cov=random_spd_matrix(rng_state_cov, shape=(shape[0], shape[0])),
    )


@case(tags=["vector", "diag-cov"])
@parametrize(
    cov=[backend.eye(7, dtype=backend.float32), linops.Scaling(2.7, shape=(20, 20))],
    ids=["backend.eye", "linops.Scaling"],
)
def case_vector_diag_cov(cov: MatrixType) -> randvars.Normal:
    rng_state = tests.utils.random.rng_state_from_sampling_args(
        base_seed=12390,
        shape=cov.shape,
        dtype=cov.dtype,
    )

    return randvars.Normal(
        mean=3.1 * backend.random.standard_normal(rng_state, shape=cov.shape[0]),
        cov=cov,
    )


@case(tags=["degenerate", "constant", "vector"])
@parametrize(
    cov=[backend.zeros, linops.Zero], ids=["cov=backend.zeros", "cov=linops.Zero"]
)
@parametrize(shape=[(3,)])
def case_vector_zero_cov(cov: MatrixType, shape: ShapeType) -> randvars.Normal:
    rng_state_mean = tests.utils.random.rng_state_from_sampling_args(
        base_seed=624,
        shape=shape,
    )
    mean = backend.random.standard_normal(shape=shape, rng_state=rng_state_mean)
    return randvars.Normal(mean=mean, cov=cov(shape=2 * shape))


@case(tags=["matrix"])
@parametrize(shape=[(1, 1), (5, 1), (1, 4), (2, 2), (3, 4)])
def case_matrix(shape: ShapeType) -> randvars.Normal:
    rng_state_mean, rng_state_cov = backend.random.split(
        tests.utils.random.rng_state_from_sampling_args(
            base_seed=453987,
            shape=shape,
        ),
        num=2,
    )

    return randvars.Normal(
        mean=4.0 * backend.random.standard_normal(rng_state_mean, shape=shape),
        cov=random_spd_matrix(
            rng_state_cov, shape=(shape[0] * shape[1], shape[0] * shape[1])
        ),
    )


@case(tags=["matrix", "mean-op", "cov-op"])
@parametrize(shape=[(1, 1), (2, 1), (1, 3), (2, 2)])
def case_matrix_mean_op_kronecker_cov(shape: ShapeType) -> randvars.Normal:
    rng_state_mean, rng_state_cov_A, rng_state_cov_B = backend.random.split(
        tests.utils.random.rng_state_from_sampling_args(
            base_seed=421376,
            shape=shape,
        ),
        num=3,
    )

    cov = linops.Kronecker(
        A=random_spd_matrix(rng_state_cov_A, shape=(shape[0], shape[0])),
        B=random_spd_matrix(rng_state_cov_B, shape=(shape[1], shape[1])),
    )
    cov.is_symmetric = True
    cov.A.is_symmetric = True
    cov.B.is_symmetric = True

    return randvars.Normal(
        mean=linops.aslinop(
            backend.random.standard_normal(rng_state_mean, shape=shape)
        ),
        cov=cov,
    )


@case(tags=["degenerate", "constant", "matrix", "cov-op"])
@parametrize(shape=[(2, 3)])
def case_matrix_zero_cov(shape: ShapeType) -> randvars.Normal:
    rng_state_mean = tests.utils.random.rng_state_from_sampling_args(
        base_seed=624,
        shape=shape,
    )
    mean = backend.random.standard_normal(shape=shape, rng_state=rng_state_mean)
    cov = linops.Kronecker(
        linops.Zero(shape=(shape[0], shape[0])), linops.Zero(shape=(shape[1], shape[1]))
    )
    return randvars.Normal(mean=mean, cov=cov)
