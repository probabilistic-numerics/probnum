"""Policies for Bayesian quadrature."""

from ._max_aquisition_policy import MaxAcquisitionPolicy, RandomMaxAcquisitionPolicy
from ._policy import Policy
from ._random_policy import RandomPolicy
from ._van_der_corput_policy import VanDerCorputPolicy

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "Policy",
    "RandomPolicy",
    "VanDerCorputPolicy",
    "MaxAcquisitionPolicy",
    "RandomMaxAcquisitionPolicy",
]

# Set correct module paths. Corrects links and module paths in documentation.
Policy.__module__ = "probnum.quad.solvers.policies"
RandomPolicy.__module__ = "probnum.quad.solvers.policies"
VanDerCorputPolicy.__module__ = "probnum.quad.solvers.policies"
RandomMaxAcquisitionPolicy.__module__ = "probnum.quad.solvers.policies"
MaxAcquisitionPolicy.__module__ = "probnum.quad.solvers.policies"
