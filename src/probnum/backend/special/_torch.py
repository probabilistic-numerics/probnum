"""Special functions in PyTorch."""

try:
    import torch
    from torch.special import ndtr, ndtri
except ModuleNotFoundError:
    pass


def gamma(x: torch.Tensor, /) -> torch.Tensor:
    raise NotImplementedError


def modified_bessel2(x: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    return NotImplementedError
