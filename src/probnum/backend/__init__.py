"""Generic computation backend.

ProbNum's backend implements a unified API for computations with arrays / tensors, that
allows writing generic code and the use of a custom backend library (currently NumPy,
JAX and PyTorch).

.. note ::

    The interface provided by this module follows the `Python array API standard
    <https://data-apis.org/array-api/latest/index.html>`_ closely, which defines a
    common API for array and tensor Python libraries.
"""

from __future__ import annotations

import inspect
import sys

from ._select import Backend, select_backend as _select_backend

BACKEND = _select_backend()

# isort: off

from ._dispatcher import Dispatcher

from ._array_object import *
from ._data_types import *
from ._core import *
from ._constants import *
from ._control_flow import *
from ._creation_functions import *
from ._elementwise_functions import *
from ._manipulation_functions import *
from ._searching_functions import *
from ._sorting_functions import *
from ._statistical_functions import *


from . import (
    _array_object,
    _data_types,
    _core,
    _constants,
    _control_flow,
    _creation_functions,
    _elementwise_functions,
    _manipulation_functions,
    _searching_functions,
    _sorting_functions,
    _statistical_functions,
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
    + _searching_functions.__all__
    + _sorting_functions.__all__
    + _statistical_functions.__all__
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
# seems to not work for our doc build setup.
__all__.sort()

# Set correct module paths. Corrects links and module paths in documentation.
member_dict = dict(inspect.getmembers(sys.modules[__name__]))
for member_name in __all__imported_modules:
    if member_name == "Array" or member_name == "Scalar":
        continue  # Avoids overriding the __module__ of aliases, which can cause bugs.

    try:
        member_dict[member_name].__module__ = "probnum.backend"
    except (AttributeError, TypeError):
        pass
