from typing import Optional, Tuple, Union

import torch
from torch.linalg import qr


def norm(
    x: torch.Tensor,
    ord: Optional[Union[int, str]] = None,
    axis: Optional[Tuple[int, ...]] = None,
    keepdims: bool = False,
):
    return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def cholesky(
    a: torch.Tensor,
    *,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
):
    return torch.linalg.cholesky(a, upper=not lower)


def solve_triangular(
    A: torch.Tensor,
    b: torch.Tensor,
    *,
    transpose: bool = False,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> torch.Tensor:
    if b.ndim == 1:
        return torch.triangular_solve(
            b[:, None],
            A,
            upper=not lower,
            transpose=transpose,
            unitriangular=unit_diagonal,
        ).solution[:, 0]

    return torch.triangular_solve(
        b,
        A,
        upper=not lower,
        transpose=transpose,
        unitriangular=unit_diagonal,
    ).solution


def solve_cholesky(
    cholesky: torch.Tensor,
    b: torch.Tensor,
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
):
    if b.ndim == 1:
        return torch.cholesky_solve(b[:, None], cholesky, upper=not lower)[:, 0]

    return torch.cholesky_solve(b, cholesky, upper=not lower)
