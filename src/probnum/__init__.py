"""
ProbNum
=====
Probabilistic Numerics in Python.
  1. Linear Algebra
  2. Quadrature
  3. Differential Equations
  4. Bayesian filtering and smoothing
and more.


Available subpackages
---------------------
prob
    Random variables, distributions and sampling.
linalg
    Probabilistic numerical linear algebra.
quad
    Bayesian quadrature / numerical integration.
filtsmooth
    Bayesian filtering and smoothing.
diffeq
    Probabilistic solvers for ordinary differential equations.
"""
# pylint: disable=wrong-import-order

from . import diffeq
from . import filtsmooth
from . import linalg
from . import quad
from . import random_variables
from . import utils

from .random_variables import asrandvar, RandomVariable

# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
