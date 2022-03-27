import numpy as np

from . import _core


def assert_allclose(actual, desired, *args, **kwargs):
    np.testing.assert_allclose(
        *_core.to_numpy(actual, desired),
        *args,
        **kwargs,
    )


def assert_array_equal(x, y, *args, **kwargs):
    np.testing.assert_array_equal(
        *_core.to_numpy(x, y),
        *args,
        **kwargs,
    )
