"""Utilities for kernel tests"""

from typing import Optional, Tuple


def _shape_param_to_id_str(shape_param: Optional[Tuple[Optional[int], ...]]):
    """Convert kernel input shape parameter used in `test_call.py` and `conftest.py`
    into a human readable representation which is used in the pytest parameter id."""

    if shape_param is None:
        return "None"

    shape_strs = tuple("indim" if dim is None else str(dim) for dim in shape_param)

    if len(shape_strs) == 1:
        return f"({shape_strs[0]},)"

    return f"({', '.join(shape_strs)})"
