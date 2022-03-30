"""Numerical constants."""

import numpy as np

from .._creation_functions import asarray
from ..typing import Scalar

__all__ = ["inf", "nan", "e", "pi"]

nan: Scalar = asarray(np.nan)
inf: Scalar = asarray(np.inf)
e: Scalar = asarray(np.e)
pi: Scalar = asarray(np.pi)
