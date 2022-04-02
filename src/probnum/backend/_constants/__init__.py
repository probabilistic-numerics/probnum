"""Numerical constants."""

import numpy as np

from .._creation_functions import asarray
from ..typing import Scalar

__all__ = ["inf", "nan", "e", "pi"]

nan: Scalar = asarray(np.nan)
"""IEEE 754 floating-point representation of Not a Number (``NaN``)."""

inf: Scalar = asarray(np.inf)
"""IEEE 754 floating-point representation of (positive) infinity."""

e: Scalar = asarray(np.e)
"""IEEE 754 floating-point representation of Euler's constant.

``e = 2.71828182845904523536028747135266249775724709369995...``
"""

pi: Scalar = asarray(np.pi)
"""IEEE 754 floating-point representation of the mathematical constant ``π``.

``pi = 3.1415926535897932384626433...``
"""
