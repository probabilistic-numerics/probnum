from pytest_cases import case, parametrize

from probnum import backend, linops, randvars
from probnum.backend.typing import ShapeType
from probnum.problems.zoo.linalg import random_spd_matrix
import tests.utils


@case(tags=["symmetric-matrix"])
@parametrize("shape", [(1, 1), (2, 2), (3, 3), (5, 5)])
def case_symmetric_matrix(shape: ShapeType) -> randvars.SymmetricMatrixNormal:
    seed_mean, seed_cov = backend.random.split(
        tests.utils.random.seed_from_sampling_args(
            base_seed=453987,
            shape=shape,
        ),
        num=2,
    )

    assert shape[0] == shape[1]

    return randvars.SymmetricMatrixNormal(
        mean=random_spd_matrix(seed_mean, shape[0]),
        cov=linops.SymmetricKronecker(random_spd_matrix(seed_cov, shape[0])),
    )
