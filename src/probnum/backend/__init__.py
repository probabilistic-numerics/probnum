"""Generic computation backend."""

from ._select import Backend, select_backend as _select_backend

BACKEND = _select_backend()

# isort: off

from ._dispatcher import Dispatcher

from ._core import *

from . import (
    _core,
    autodiff,
    linalg,
    random,
    special,
    _elementwise_functions,
    _manipulation_functions,
    _creation_functions,
    _constants,
)

# isort: on

__all__ = (
    [
        "Backend",
        "BACKEND",
        "Dispatcher",
    ]
    + _core.__all__
    + sum(
        [
            module.__all__
            for module in [
                _elementwise_functions,
                _manipulation_functions,
                _creation_functions,
                _constants,
            ]
        ]
    )
)
