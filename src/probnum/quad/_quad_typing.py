from typing import Tuple, Union

import numpy as np

from probnum.typing import FloatLike

_DomainType = Tuple[np.ndarray, np.ndarray]
DomainLike = Union[Tuple[FloatLike, FloatLike], _DomainType]
