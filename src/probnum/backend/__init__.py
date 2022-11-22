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

import builtins
import inspect
import sys

# isort: off

from ._dispatcher import Dispatcher

from ._array_object import *
from ._data_types import *
from ._constants import *
from ._control_flow import *
from ._creation_functions import *
from ._elementwise_functions import *
from ._logic_functions import *
from ._manipulation_functions import *
from ._searching_functions import *
from ._sorting_functions import *
from ._statistical_functions import *
from ._jit_compilation import *
from ._vectorization import *

from . import (
    _array_object,
    _data_types,
    _constants,
    _control_flow,
    _creation_functions,
    _elementwise_functions,
    _logic_functions,
    _manipulation_functions,
    _searching_functions,
    _sorting_functions,
    _statistical_functions,
    _jit_compilation,
    _vectorization,
    autodiff,
    linalg,
    random,
    special,
)

# isort: on

# Import some often used functions into probnum.backend
from .linalg import diagonal, einsum, matmul, outer, tensordot, vecdot

# Define probnum.backend API
__all__imported_modules = (
    _array_object.__all__
    + _data_types.__all__
    + _constants.__all__
    + _control_flow.__all__
    + _creation_functions.__all__
    + _elementwise_functions.__all__
    + _logic_functions.__all__
    + _manipulation_functions.__all__
    + _searching_functions.__all__
    + _sorting_functions.__all__
    + _statistical_functions.__all__
    + _jit_compilation.__all__
    + _vectorization.__all__
)
__all__ = [
    "Dispatcher",
] + __all__imported_modules

# Set correct module paths. Corrects links and module paths in documentation.
member_dict = dict(inspect.getmembers(sys.modules[__name__]))
for member_name in __all__imported_modules:
    if builtins.any([member_name == mn for mn in ["Array", "Scalar", "Device"]]):
        continue  # Avoids overriding the __module__ of aliases, which can cause bugs.

    try:
        member_dict[member_name].__module__ = "probnum.backend"
    except (AttributeError, TypeError):
        pass
