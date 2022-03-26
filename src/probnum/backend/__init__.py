"""Generic computation backend."""
import inspect
import sys

from ._select import Backend, select_backend as _select_backend

BACKEND = _select_backend()

# isort: off

from ._dispatcher import Dispatcher

from ._core import *
from ._array_object import *
from ._constants import *
from ._creation_functions import *
from ._elementwise_functions import *
from ._manipulation_functions import *

from . import (
    _core,
    _array_object,
    _constants,
    _creation_functions,
    _elementwise_functions,
    _manipulation_functions,
    autodiff,
    linalg,
    random,
    special,
)

# isort: on

__all__imported_modules = sum(
    [
        module.__all__
        for module in [
            _array_object,
            _constants,
            _creation_functions,
            _elementwise_functions,
            _manipulation_functions,
        ]
    ]
)
__all__ = (
    [
        "Backend",
        "BACKEND",
        "Dispatcher",
    ]
    + _core.__all__
    + __all__imported_modules
)

# Set correct module paths. Corrects links and module paths in documentation.
member_dict = dict(inspect.getmembers(sys.modules[__name__]))
for member_name in __all__imported_modules:
    try:
        member_dict[member_name].__module__ = "probnum.backend"
    except TypeError:
        pass
