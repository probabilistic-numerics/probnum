import numpy as np
import pytest

from probnum import filtsmooth


@pytest.fixture
def state_list():
    return [
        filtsmooth.ParticleFilterState(
            weights=np.ones(10) / 10, particles=np.random.rand(10, 2)
        )
        for _ in range(20)
    ]


@pytest.fixture
def posterior(state_list):
    return filtsmooth.ParticleFilterPosterior(state_list, locations=np.random.rand(20))


def test_len(posterior):
    assert len(posterior) == 20


def test_getitem(posterior):
    assert len(posterior) == 20


def test_call(posterior):
    with pytest.raises(NotImplementedError):
        posterior(0.0)
