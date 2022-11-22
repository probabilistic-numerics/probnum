from typing import Callable, Union

from probnum import backend, randvars
from probnum.backend.typing import ShapeType
from probnum.problems.zoo.linalg import random_spd_matrix

import tests.utils

GeneratorFnType = Callable[[ShapeType], Union[randvars.RandomVariable, backend.Array]]


def array_generator(shape: ShapeType) -> backend.Array:
    return 3.0 * backend.random.standard_normal(
        tests.utils.random.rng_state_from_sampling_args(
            base_seed=561562,
            shape=shape,
        ),
        shape=shape,
    )


def constant_generator(shape: ShapeType) -> randvars.Constant:
    return randvars.Constant(array_generator(shape))


def normal_generator(shape: ShapeType) -> randvars.Normal:
    rng_state_mean, rng_state_cov = backend.random.split(
        tests.utils.random.rng_state_from_sampling_args(
            base_seed=561562,
            shape=shape,
        ),
        num=2,
    )

    mean = 5.0 * backend.random.standard_normal(rng_state_mean, shape=shape)

    return randvars.Normal(
        mean=mean,
        cov=random_spd_matrix(
            rng_state_cov, shape=() if mean.shape == () else (mean.size, mean.size)
        ),
    )
