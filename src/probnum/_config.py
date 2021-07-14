import contextlib
import copy
import dataclasses
from typing import Any


@dataclasses.dataclass
class Option:
    name: str
    default_value: Any
    description: str
    value: Any

    def __repr__(self) -> str:
        _r = "<Configuration.Option "
        _r += f"name={self.name}, value={self.value}>"
        return _r


class Configuration:
    r"""
    Configuration over which some mechanics of ProbNum can be controlled dynamically.

    ProbNum provides some configurations together with default values. These
    are listed in the tables below.
    Additionally, users can register their own configuration entries via
    :meth:`register`. Configuration entries can only be registered once and can only
    be used (accessed or overwritten) once they have been registered.

    .. probnum-config-options::

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

    def __init__(self) -> None:
        object.__setattr__(self, "_options_registry", dict())

    def __getattr__(self, key):
        if key not in self._options_registry:
            raise AttributeError(
                f"getConfiguration entry {key} does not exist yet."
                "Configuration entries must be `register`ed before they can be "
                "accessed."
            )
        return self._options_registry[key].value

    def __setattr__(self, key: str, value: Any) -> None:
        if key not in self._options_registry:
            raise AttributeError(
                f"setConfiguration entry {key} does not exist yet."
                "Configuration entries must be `register`ed before they can be "
                "accessed."
            )

        self._options_registry[key].value = value

    def __repr__(self) -> str:
        return repr(self._options_registry)

    @contextlib.contextmanager
    def __call__(self, **kwargs) -> None:
        old_entries = dict()

        for key, value in kwargs.items():
            if key not in self._options_registry:
                raise AttributeError(
                    f"Configuration entry {key} does not exist yet."
                    "Configuration entries must be `register`ed before they can be "
                    "accessed."
                )

            old_entries[key] = getattr(self, key)

            setattr(self, key, value)

        try:
            yield
        finally:
            for key, old_value in old_entries.items():
                setattr(self, key, old_value)

    def register(self, key: str, default_value: Any, description: str) -> None:
        r"""Register a new configuration option.

        Parameters
        ----------
        key:
            The name of the configuration option. This will be the ``key`` when calling
            ``with config(key=<some_value>): ...``.
        default_value:
            The default value of the configuration option.
        description:
            A short description of the configuration option and what it controls.
        """
        if key in self._options_registry:
            raise KeyError(
                f"Configuration entry {key} does already exist and "
                "cannot be registered again."
            )
        new_config_option = Option(
            name=key,
            default_value=default_value,
            description=description,
            value=default_value,
        )
        self._options_registry[key] = new_config_option


# Create a single, global configuration object,...
_GLOBAL_CONFIG_SINGLETON = Configuration()

# ... define some configuration options, and the respective default values
# (which have to be documented in the Configuration-class docstring!!), ...
_DEFAULT_CONFIG_OPTIONS = [
    # list of tuples (config_key, default_value)
    (
        "covariance_inversion_damping",
        1e-12,
        (
            "A (typically small) value that is per default added to the diagonal "
            "of covariance matrices in order to make inversion numerically stable."
        ),
    ),
]

# ... and register the default configuration options.
for key, default_value, descr in _DEFAULT_CONFIG_OPTIONS:
    _GLOBAL_CONFIG_SINGLETON.register(key, default_value, descr)
