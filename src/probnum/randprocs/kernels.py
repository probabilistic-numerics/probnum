"""This is an alias for the :mod:`probnum.randprocs.covfuncs` module. We mainly include
it and the :class:`Kernel` alias for backwards compatibility.

.. deprecated:: 0.1.23
    The module is deprecated and should not be used in new code.
    Use :mod:`probnum.randprocs.covfuncs` instead.
"""
from .covfuncs import *  # pylint: disable=wildcard-import,unused-wildcard-import

Kernel = CovarianceFunction
