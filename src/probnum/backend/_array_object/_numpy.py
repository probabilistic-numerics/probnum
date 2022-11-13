"""Array object in NumPy."""
from typing import Literal, TypeVar

from numpy import (  # pylint: disable=redefined-builtin, unused-import
    generic as Scalar,
    ndarray as Array,
)

Device = Literal["cpu"]
