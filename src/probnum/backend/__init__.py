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
)

# isort: on

__all__ = [
    "Backend",
    "BACKEND",
    "Dispatcher",
] + _core.__all__
