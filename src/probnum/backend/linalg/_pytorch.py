import torch


def cho_solve(c_and_lower, b, overwrite_b=False, check_finite=True):
    (c, lower) = c_and_lower

    if b.ndim == 1:
        return torch.cholesky_solve(b[:, None], c, upper=not lower)[:, 0]

    return torch.cholesky_solve(b, c, upper=not lower)


def cholesky(a, lower=False, overwrite_a=False, check_finite=True):
    return torch.linalg.cholesky(a, upper=not lower)
