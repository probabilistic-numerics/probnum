"""Functionality for random number generation implemented in the PyTorch backend."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.distributions.utils import broadcast_all

from probnum.backend.typing import Seed, ShapeType

RNGState = np.random.SeedSequence


def rng_state(seed: Seed) -> RNGState:
    return np.random.SeedSequence(seed)


def split(rng_state: RNGState, num: int = 2) -> Sequence[RNGState]:
    return rng_state.spawn(num)


def _rng_from_rng_state(rng_state: RNGState) -> torch.Generator:
    """Create a random generator instance initialized with the given state."""

    if not isinstance(rng_state, RNGState):
        raise TypeError(
            "`rng_state`s should always have type :class:`~backend.random.RNGState`."
        )

    rng = torch.Generator()
    return rng.manual_seed(int(rng_state.generate_state(1, dtype=np.uint64)[0]))


def uniform(
    rng_state: RNGState,
    shape: ShapeType = (),
    dtype: torch.dtype = torch.double,
    minval: torch.Tensor = torch.as_tensor(0.0),
    maxval: torch.Tensor = torch.as_tensor(1.0),
) -> torch.Tensor:
    rng = _rng_from_rng_state(rng_state)
    return (maxval - minval) * torch.rand(shape, generator=rng, dtype=dtype) + minval


def standard_normal(
    rng_state: RNGState,
    shape: ShapeType = (),
    dtype: torch.dtype = torch.double,
) -> torch.Tensor:
    rng = _rng_from_rng_state(rng_state)

    return torch.randn(shape, generator=rng, dtype=dtype)


def gamma(
    rng_state: RNGState,
    shape_param: torch.Tensor,
    scale_param: torch.Tensor = torch.as_tensor(1.0),
    shape: ShapeType = (),
    dtype=torch.double,
) -> torch.Tensor:
    rng = _rng_from_rng_state(rng_state)

    shape_param = torch.as_tensor(shape_param, dtype=dtype)
    scale_param = torch.as_tensor(scale_param, dtype=dtype)

    # Adapted version of
    # https://github.com/pytorch/pytorch/blob/afff38182457f3500c265f232310438dded0e57d/torch/distributions/gamma.py#L59-L63
    shape_param, scale_param = broadcast_all(shape_param, scale_param)

    res_shape = shape + shape_param.shape

    return torch._standard_gamma(
        shape_param.expand(res_shape), rng
    ) * scale_param.expand(res_shape)


def uniform_so_group(
    rng_state: RNGState,
    n: int,
    shape: ShapeType = (),
    dtype: torch.dtype = torch.double,
) -> torch.Tensor:
    if n == 1:
        return torch.ones(shape + (1, 1), dtype=dtype)

    omega = standard_normal(rng_state, shape=shape + (n - 1, n), dtype=dtype)

    sample = _uniform_so_group_pushforward_fn(omega.reshape((-1, n - 1, n)))

    return sample.reshape(shape + (n, n))


@torch.jit.script
def _uniform_so_group_pushforward_fn(omega: torch.Tensor) -> torch.Tensor:
    n = omega.shape[-1]

    assert omega.ndim == 3 and omega.shape[-2] == n - 1

    samples = []

    for sample_idx in range(omega.shape[0]):
        X = torch.triu(omega[sample_idx, :, :])
        X_diag = torch.diag(X)

        D = torch.where(
            X_diag != 0,
            torch.sign(X_diag),
            torch.ones((), dtype=omega.dtype),
        )

        row_norms_sq = torch.sum(X**2, dim=1)

        diag_indices = torch.arange(n - 1)
        X[diag_indices, diag_indices] = torch.sqrt(row_norms_sq) * D

        X /= torch.sqrt((row_norms_sq - X_diag**2 + torch.diag(X) ** 2) / 2.0)[
            :, None
        ]

        H = torch.eye(n, dtype=omega.dtype)

        for idx in range(n - 1):
            H -= torch.outer(H @ X[idx, :], X[idx, :])

        D = torch.cat(
            (
                D,
                (-1.0 if n % 2 == 0 else 1.0) * torch.prod(D, dim=0, keepdim=True),
            ),
            dim=0,
        )

        samples.append(D[:, None] * H)

    return torch.stack(samples, dim=0)
