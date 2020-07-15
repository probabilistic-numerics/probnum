"""
ProbNum
=====
Probabilistic Numerics routines in Python.
  1. Linear Algebra
  2. Differential Equations
  3. Bayesian filtering and smoothing
and more.


Available subpackages
---------------------
linalg
    Probabilistic numerical linear algebra
prob
    Random variables and distributions.
filtsmooth
    Bayesian filtering and smoothing
diffeq
    Probabilistic solvers for ordinary differential equation

"""

from . import diffeq
from . import filtsmooth
from . import linalg
from . import prob
from . import quad
from . import utils


# What does the block below do? ###########################

# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
