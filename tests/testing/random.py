from collections.abc import Hashable

from probnum import backend
from probnum.typing import SeedType


def seed_from_args(*args: Hashable) -> SeedType:
    return backend.random.seed(abs(sum(map(hash, args))))
