"""Generic computation backend."""
import inspect
import sys

from ._select import Backend, select_backend as _select_backend

BACKEND = _select_backend()

# isort: off

from ._dispatcher import Dispatcher

from ._core import *
from ._data_types import *
from ._array_object import *
from ._constants import *
from ._control_flow import *
from ._creation_functions import *
from ._elementwise_functions import *
from ._manipulation_functions import *
from ._sorting_functions import *

from . import (
    _data_types,
    _array_object,
    _core,
    _constants,
    _control_flow,
    _creation_functions,
    _elementwise_functions,
    _manipulation_functions,
    _sorting_functions,
    autodiff,
    linalg,
    random,
    special,
)

# isort: on

__all__imported_modules = (
    _array_object.__all__
    + _data_types.__all__
    + _constants.__all__
    + _control_flow.__all__
    + _creation_functions.__all__
    + _elementwise_functions.__all__
    + _manipulation_functions.__all__
    + _sorting_functions.__all__
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
# Sort entries in documentation. Necessary since autodoc config option `member_order`
# seems to have no effect.
__all__.sort()

# Set correct module paths. Corrects links and module paths in documentation.
member_dict = dict(inspect.getmembers(sys.modules[__name__]))
for member_name in __all__imported_modules:
    try:
        member_dict[member_name].__module__ = "probnum.backend"
    except (AttributeError, TypeError):
        pass
