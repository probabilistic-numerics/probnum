"""/sorting functions for NumPy arrays."""
import numpy as np
from numpy import isnan  # pylint: disable=redefined-builtin, unused-import


def sort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> np.ndarray:
    kind = "quicksort"
    if stable:
        kind = "stable"

    sorted_array = np.sort(x, axis=axis, kind=kind)

    if descending:
        return np.flip(sorted_array, axis=axis)

    return sorted_array


def argsort(
    x: np.ndarray,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> np.ndarray:
    kind = "quicksort"
    if stable:
        kind = "stable"

    sort_idx = np.argsort(x, axis=axis, kind=kind)

    if descending:
        return np.flip(sort_idx, axis=axis)

    return sort_idx
