"""Types specific to the quad package."""

from typing import Tuple, Union

import numpy as np

from probnum.backend.typing import FloatLike

DomainType = Tuple[np.ndarray, np.ndarray]
DomainLike = Union[Tuple[FloatLike, FloatLike], DomainType]
