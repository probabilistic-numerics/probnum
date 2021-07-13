import contextlib
from typing import Any


class Configuration:
    """
    Configuration over which some mechanics of ProbNum can be controlled dynamically.

    ProbNum provides some configurations together with default values. These
    are listed in the tables below.
    Additionally, users can register their own configuration entries via
    :meth:`register`. Configuration entries can only be registered once and can only
    be used (accessed or overwritten) once they have been registered.

    +----------------------------------+---------------+----------------------------------------------+
    | Config entry                     | Default value | Description                                  |
    +==================================+===============+==============================================+
    | ``covariance_inversion_damping`` | ``1e-12``     | A (typically small) value that is per        |
    |                                  |               | default added to the diagonal of covariance  |
    |                                  |               | matrices in order to make inversion          |
    |                                  |               | numerically stable.                          |
    +----------------------------------+---------------+----------------------------------------------+
    | ``...``                          | ``...``       | ...                                          |
    +----------------------------------+---------------+----------------------------------------------+

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
        """Register new configuration option.

        Parameters
        ----------
        key:
            The name of the configuration option. This will be the ``key`` when calling
            ``with config(key=<some_value>): ...``.
        default_value:
            The default value of the configuration option.
        """
        if hasattr(self, key):
            raise KeyError(
                f"Configuration entry {key} does already exist and "
                "cannot be registered again."
            )
        self.__dict__[key] = default_value


# Create a single, global configuration object,...
_GLOBAL_CONFIG_SINGLETON = Configuration()

# ... define some configuration options, and the respective default values
# (which have to be documented in the Configuration-class docstring!!), ...
_DEFAULT_CONFIG_OPTIONS = [
    # list of tuples (config_key, default_value)
    ("covariance_inversion_damping", 1e-12),
]

# ... and register the default configuration options.
for key, default_value in _DEFAULT_CONFIG_OPTIONS:
    _GLOBAL_CONFIG_SINGLETON.register(key, default_value)
