from typing import Optional, Sequence

import numpy as np
import torch
from torch.distributions.utils import broadcast_all

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
    a: torch.Tensor,
    scale=1.0,
    shape=(),
    dtype=torch.double,
):
    rng = _make_rng(seed)

    a = a.to(dtype)
    scale = scale.to(dtype)

    # Adapted version of
    # https://github.com/pytorch/pytorch/blob/afff38182457f3500c265f232310438dded0e57d/torch/distributions/gamma.py#L59-L63
    a, scale = broadcast_all(a, scale)

    res_shape = shape + a.shape

    return torch._standard_gamma(a.expand(res_shape), rng) * scale.expand(res_shape)


def _make_rng(seed: np.random.SeedSequence) -> torch.Generator:
    rng = torch.Generator()

    # state = seed.generate_state(_RNG_STATE_SIZE // 4, dtype=np.uint32)
    # rng.set_state(torch.ByteTensor(state.view(np.uint8)))

    rng.manual_seed(int(seed.generate_state(1, dtype=np.uint64)[0]))

    return rng
