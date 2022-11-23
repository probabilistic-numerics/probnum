"""Special functions in NumPy / SciPy."""
import numpy as np
from scipy.special import gamma, kv, ndtr, ndtri  # pylint: disable=unused-import


def modified_bessel(x: np.ndarray, order: float) -> np.ndarray:
    return kv(order, x)
