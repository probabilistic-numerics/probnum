import abc

import numpy as np
import pytest

import probnum
from probnum import randvars


class _TestConfig(abc.ABC):
    @abc.abstractmethod
    @pytest.fixture(autouse=True)
    def _setup(self):
        pass

    def test_contextmanager(self):
        none_vals = {key: None for (key, _) in self.defaults}

        for key, default_val in self.defaults:
            # Check if default is correct before context manager
            assert getattr(probnum.config, key) == default_val
            # Temporarily set all config values to None
            with probnum.config(**none_vals):
                assert getattr(probnum.config, key) is None

            # Check if the original (default) values are set after exiting the context
            # manager
            assert getattr(probnum.config, key) == default_val


class TestRandvarConfig(_TestConfig):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.defaults = [
            ("covariance_inversion_damping", 1e-12),
        ]

    @pytest.fixture
    def zero_cov_normal(self):
        return randvars.Normal(np.random.rand(5), np.zeros((5, 5)))

    def test_randvars_config(self, zero_cov_normal):
        chol_1 = zero_cov_normal.dense_cov_cholesky()
        np.testing.assert_allclose(
            np.diag(chol_1),
            np.full(shape=(chol_1.shape[0],), fill_value=np.sqrt(1e-12)),
        )

        with probnum.config(covariance_inversion_damping=1e-3):
            chol_2 = zero_cov_normal.dense_cov_cholesky()
            np.testing.assert_allclose(
                np.diag(chol_2),
                np.full(shape=(chol_1.shape[0],), fill_value=np.sqrt(1e-3)),
            )


def test_register():
    # Check if registering a new config entry works
    probnum.config.register("some_config", 3.14)
    assert probnum.config.some_config == 3.14

    # When registering a new entry with an already existing name, throw
    with pytest.raises(KeyError):
        probnum.config.register("some_config", 4.2)

    # Check if temporarily setting the config entry to a different value (via
    # the context manager) works
    with probnum.config(some_config=9.9):
        assert probnum.config.some_config == 9.9

    # Upon exiting the context manager, the previous value is restored
    assert probnum.config.some_config == 3.14

    # Setting the config entry permanently also works by
    # accessing the attribute directly
    probnum.config.some_config = 4.5
    assert probnum.config.some_config == 4.5

    # Setting a config entry before registering it, does not work. Neither via
    # the context manager ...
    with pytest.raises(KeyError):
        with probnum.config(unknown_config=False):
            pass

    # ... nor by accessing the attribute directly.
    with pytest.raises(KeyError):
        probnum.config.unknown_config = False
