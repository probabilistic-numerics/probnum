"""Implementation of linear algebra functionality in PyTorch."""

from typing import Literal, Optional, Tuple, Union

try:
    import torch

    # pylint: disable=unused-import
    from torch import diagonal, kron, matmul, outer, tensordot
    from torch.linalg import (
        det,
        eigh,
        eigvalsh,
        inv,
        matrix_power,
        matrix_rank,
        pinv,
        qr,
        slogdet,
        solve,
        svd,
        svdvals,
        vecdot,
    )
except ModuleNotFoundError:
    pass


def matrix_transpose(x: "torch.Tensor", /) -> "torch.Tensor":
    return torch.transpose(x, -2, -1)


def trace(x: "torch.Tensor", /, *, offset: int = 0) -> "torch.Tensor":
    if offset != 0:
        raise NotImplementedError

    return torch.trace(x)


def pinv(
    x: "torch.Tensor", rtol: Optional[Union[float, "torch.Tensor"]] = None
) -> "torch.Tensor":
    return torch.linalg.pinv(x, rtol=rtol)


def einsum(
    *arrays: "torch.Tensor",
    optimization: Optional[str] = "greedy",
):
    return torch.einsum(*arrays)


def vector_norm(
    x: "torch.Tensor",
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ord: Union[int, float, Literal["inf", "-inf"]] = 2,
) -> "torch.Tensor":
    return torch.linalg.vector_norm(x, ord=ord, dim=axis, keepdim=keepdims)


def matrix_norm(
    x: "torch.Tensor", /, *, keepdims: bool = False, ord="fro"
) -> "torch.Tensor":
    return torch.linalg.matrix_norm(x, ord=ord, dim=(-2, -1), keepdim=keepdims)


def norm(
    x: "torch.Tensor",
    ord: Optional[Union[int, str]] = None,
    axis: Optional[Tuple[int, ...]] = None,
    keepdims: bool = False,
):
    return torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def cholesky(x: "torch.Tensor", /, *, upper: bool = False) -> "torch.Tensor":
    try:
        return torch.linalg.cholesky(x, upper=upper)
    except RuntimeError:
        return (torch.triu if upper else torch.tril)(torch.full_like(x, float("nan")))


def solve_triangular(
    A: "torch.Tensor",
    b: "torch.Tensor",
    *,
    transpose: bool = False,
    lower: bool = False,
    unit_diagonal: bool = False,
) -> "torch.Tensor":
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
    cholesky: "torch.Tensor",
    b: "torch.Tensor",
    *,
    lower: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
):
    if b.ndim == 1:
        return torch.cholesky_solve(b[:, None], cholesky, upper=not lower)[:, 0]

    return torch.cholesky_solve(b, cholesky, upper=not lower)


def qr(
    x: "torch.Tensor", /, *, mode: Literal["reduced", "complete", "r"] = "reduced"
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    q, r = torch.linalg.qr(x, mode=mode)
    return q, r
