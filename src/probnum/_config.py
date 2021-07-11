import contextlib
from typing import Any


class Configuration:
    """
    >>> import probnum
    >>> probnum.config.covariance_inversion_damping
    1e-12
    >>> with probnum.config(
    ...     covariance_inversion_damping=1e-2,
    ... ):
    ...     probnum.config.covariance_inversion_damping
    0.01
    """

    @contextlib.contextmanager
    def __call__(self, **kwargs) -> None:
        old_entries = dict()

        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise KeyError(
                    f"Configuration entry {key} does not exist yet."
                    "Configuration entries must be `register`ed before they can be "
                    "accessed."
                )

            old_entries[key] = getattr(self, key)

            setattr(self, key, value)

        try:
            yield
        finally:
            self.__dict__.update(old_entries)

    def __setattr__(self, key: str, value: Any) -> None:
        if not hasattr(self, key):
            raise KeyError(
                f"Configuration entry {key} does not exist yet."
                "Configuration entries must be `register`ed before they can be "
                "accessed."
            )

        self.__dict__[key] = value

    def register(self, key: str, default_value: Any, docstring: str) -> None:
        if hasattr(self, key):
            raise KeyError(
                f"Configuration entry {key} does already exist and "
                "cannot be registered again."
            )
        self.__dict__[key] = default_value
