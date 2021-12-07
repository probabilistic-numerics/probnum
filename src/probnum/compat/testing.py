import numpy as np

from . import _core


def assert_allclose(actual, desired, *args, **kwargs):
    actual = _core.to_numpy(actual)
    desired = _core.to_numpy(desired)

    np.testing.assert_allclose(actual, desired, *args, **kwargs)
