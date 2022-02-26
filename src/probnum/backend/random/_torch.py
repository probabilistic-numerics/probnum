from typing import Optional, Sequence

import numpy as np
import torch
from torch.distributions.utils import broadcast_all

from probnum.typing import DTypeLike, ShapeLike

_RNG_STATE_SIZE = torch.Generator().get_state().shape[0]


def seed(seed: Optional[int]) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed)


def split(
    seed: np.random.SeedSequence, num: int = 2
) -> Sequence[np.random.SeedSequence]:
    return seed.spawn(num)


def standard_normal(seed: np.random.SeedSequence, shape=(), dtype=torch.double):
    rng = _make_rng(seed)

    return torch.randn(shape, generator=rng, dtype=dtype)


def gamma(
    seed: np.random.SeedSequence,
    shape_param: torch.Tensor,
    scale_param=1.0,
    shape=(),
    dtype=torch.double,
):
    rng = _make_rng(seed)

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
    seed: np.random.SeedSequence,
    n: int,
    shape: ShapeLike = (),
    dtype: DTypeLike = torch.double,
) -> torch.Tensor:
    if n == 1:
        return torch.ones(shape + (1, 1), dtype=dtype)

    omega = standard_normal(seed, shape=shape + (n - 1, n), dtype=dtype)

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


def _make_rng(seed: np.random.SeedSequence) -> torch.Generator:
    rng = torch.Generator()

    # state = seed.generate_state(_RNG_STATE_SIZE // 4, dtype=np.uint32)
    # rng.set_state(torch.ByteTensor(state.view(np.uint8)))

    return rng.manual_seed(int(seed.generate_state(1, dtype=np.int64)[0]))
