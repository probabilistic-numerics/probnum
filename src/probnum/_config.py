"""ProbNum library configuration"""

import contextlib
import dataclasses
from typing import Any


class Configuration:
    r"""Configuration by which some mechanics of ProbNum can be controlled dynamically.

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

    _NON_REGISTERED_KEY_ERR_MSG = (
        'Configuration option "%s" does not exist yet. '
        "Configuration options must be `register`ed before they can be "
        "accessed."
    )

    @dataclasses.dataclass
    class Option:
        """Representation of a single configuration option as a key-value pair with a
        default value and a description string for documentation purposes."""

        name: str
        default_value: Any
        description: str
        value: Any

        def __repr__(self) -> str:
            _r = "<Configuration.Option "
            _r += f"name={self.name}, value={self.value}>"
            return _r

    def __init__(self) -> None:
        # This is the equivalent of `self._options_registry = dict()`.
        # After rewriting the `__setattr__` method, we have to fall back on the
        # `__setattr__` method of the super class.
        object.__setattr__(self, "_options_registry", {})

    def __getattr__(self, key: str) -> Any:
        if key not in self._options_registry:
            raise AttributeError(f'Configuration option "{key}" does not exist.')
        return self._options_registry[key].value

    def __setattr__(self, key: str, value: Any) -> None:
        if key not in self._options_registry:
            raise AttributeError(Configuration._NON_REGISTERED_KEY_ERR_MSG % key)

        self._options_registry[key].value = value

    def __repr__(self) -> str:
        return repr(self._options_registry)

    @contextlib.contextmanager
    def __call__(self, **kwargs) -> None:
        """Context manager used to set values of registered config options."""
        old_options = {}

        for key, value in kwargs.items():
            if key not in self._options_registry:
                raise AttributeError(Configuration._NON_REGISTERED_KEY_ERR_MSG % key)

            old_options[key] = self._options_registry[key].value

            self._options_registry[key].value = value

        try:
            yield
        finally:
            for key, old_value in old_options.items():
                self._options_registry[key].value = old_value

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

        Raises
        ------
        KeyError
            If the configuration option already exists.
        """
        if key in self._options_registry:
            raise KeyError(
                f"Configuration option {key} does already exist and "
                "cannot be registered again."
            )
        new_config_option = Configuration.Option(
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
    (
        "matrix_free",
        False,
        (
            r"If :obj:`True`, wherever possible, :class:`~.linops.LinearOperator`\ s "
            r"are used instead of arrays. :class:`~.linops.LinearOperator`\ s "
            r"define a matrix-vector product implicitly without instantiating the full "
            r"matrix in memory. This makes them memory- and runtime-efficient for "
            r"linear algebra operations."
        ),
    ),
    (
        "lazy_matrix_matrix_matmul",
        True,
        (
            r"If this is set to :obj:`False`, the matrix multiplication operator ``@`` "
            r"applied to two :class:`~probnum.linops.LinearOperator`\ s of type "
            r":class:`~probnum.linops.Matrix` multiplies the two matrices immediately "
            r"and returns the product as a :class:`~probnum.linops.Matrix`. Otherwise, "
            r"i.e. if this option is set to :obj:`True`, a :class:`~probnum.linops."
            r"ProductLinearOperator`, representing the matrix product, is returned. "
            r"Multiplying a vector with the :class:`~probnum.linops."
            r"ProductLinearOperator` is often more efficient than computing the "
            r"full matrix-matrix product first. This is why this option is set to "
            r":obj:`True` by default."
        ),
    ),
]

# ... and register the default configuration options.
def _register_defaults():
    for key, default_value, descr in _DEFAULT_CONFIG_OPTIONS:
        _GLOBAL_CONFIG_SINGLETON.register(key, default_value, descr)


_register_defaults()
