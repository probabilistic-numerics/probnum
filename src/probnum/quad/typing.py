"""Types specific to the quad package."""

from typing import Tuple, Union

import numpy as np

from probnum.typing import FloatLike

__all__ = ["DomainLike", "DomainType"]

DomainType = Tuple[np.ndarray, np.ndarray]
"""Type defining an integration domain."""

DomainLike = Union[Tuple[FloatLike, FloatLike], DomainType]
"""Object that can be converted to an integration domain."""
