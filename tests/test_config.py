import pytest

import probnum
from probnum._config import _DEFAULT_CONFIG_OPTIONS


def test_defaults():
    none_vals = {key: None for (key, _, _) in _DEFAULT_CONFIG_OPTIONS}

    for key, default_val, _ in _DEFAULT_CONFIG_OPTIONS:
        # Check if default is correct before context manager
        assert getattr(probnum.config, key) == default_val
        # Temporarily set all config values to None
        with probnum.config(**none_vals):
            assert getattr(probnum.config, key) is None

        # Check if the original (default) values are set after exiting the context
        # manager
        assert getattr(probnum.config, key) == default_val


def test_register():
    # Check if registering a new config entry works
    probnum.config.register("some_config", 3.14, "Dummy description.")
    assert hasattr(probnum.config, "some_config")
    assert probnum.config.some_config == 3.14

    # When registering a new entry with an already existing name, throw
    with pytest.raises(KeyError):
        probnum.config.register("some_config", 4.2, "Dummy description.")

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
    with pytest.raises(AttributeError):
        with probnum.config(unknown_config=False):
            pass

    # ... nor by accessing the attribute directly.
    with pytest.raises(AttributeError):
        probnum.config.unknown_config = False
