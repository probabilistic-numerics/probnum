"""Torch tensor creation functions."""

import torch


def tril(x: torch.Tensor, /, k: int = 0) -> torch.Tensor:
    return torch.tril(x, diagonal=k)


def triu(x: torch.Tensor, /, k: int = 0) -> torch.Tensor:
    return torch.triu(x, diagonal=k)
