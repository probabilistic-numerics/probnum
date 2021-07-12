import contextlib
from typing import Any


class Configuration:
    """
    Configuration over which some mechanics of probnum can be controlled dynamically.

    ``probnum`` provides some configurations together with default values. These
    are listed in the tables below.
    Additionally, the user can register own configuration entries via the method
    :meth:`register`. Configuration entries can only be registered once and can only
    be used (accessed or overwritten) once they have been registered.

    - ``probnum.randvars``

    +----------------------------------+-----------------------------------------------+
    | Config entry                     | Description                                   |
    +==================================+===============================================+
    | ``covariance_inversion_damping`` | A (typically small) value that is per         |
    |                                  | default added to the diagonal of covariance   |
    |                                  | matrices in order to make inversion           |
    |                                  | numerically stable.                           |
    +----------------------------------+-----------------------------------------------+
    | ``...``                          | ...                                           |
    +----------------------------------+-----------------------------------------------+

    - ``probnum.diffeq``

    - ``...``


    Examples
    ========

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

    def register(self, key: str, default_value: Any) -> None:
        if hasattr(self, key):
            raise KeyError(
                f"Configuration entry {key} does already exist and "
                "cannot be registered again."
            )
        self.__dict__[key] = default_value
