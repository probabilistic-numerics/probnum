from typing import Tuple, Union

import numpy as np

from probnum.typing import FloatLike

DomainLike = Union[Tuple[FloatLike, FloatLike], Tuple[np.ndarray, np.ndarray]]
_DomainType = Tuple[np.ndarray, np.ndarray]
