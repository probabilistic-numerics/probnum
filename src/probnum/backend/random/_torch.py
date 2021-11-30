from typing import Optional, Sequence

import numpy as np
import torch

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


def _make_rng(seed: np.random.SeedSequence) -> torch.Generator:
    rng = torch.Generator()

    # state = seed.generate_state(_RNG_STATE_SIZE // 4, dtype=np.uint32)
    # rng.set_state(torch.ByteTensor(state.view(np.uint8)))

    rng.manual_seed(int(seed.generate_state(1, dtype=np.uint64)[0]))

    return rng
