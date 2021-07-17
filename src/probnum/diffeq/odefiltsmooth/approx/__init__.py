"""Approximate information operators."""

from ._approx import ApproximationStrategy
from ._ek import EK0, EK1

__all__ = ["ApproximationStrategy", "EK0", "EK1"]


# Set correct module paths (for superclasses).
# Corrects links and module paths in documentation.
ApproximationStrategy.__module__ = "probnum.diffeq.odefiltsmooth.approx"
EK0.__module__ = "probnum.diffeq.odefiltsmooth.approx"
EK1.__module__ = "probnum.diffeq.odefiltsmooth.approx"
