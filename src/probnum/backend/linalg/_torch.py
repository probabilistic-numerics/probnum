import torch


def cholesky(
    a: torch.Tensor,
    *,
    lower: bool = False,
    overwrite_a: bool = False,
    check_finite: bool = True,
):
    return torch.linalg.cholesky(a, upper=not lower)


def cholesky_solve(
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
