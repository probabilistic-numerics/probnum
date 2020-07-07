"""
Random Processes

This submodule implements different types of random processes defined as collections of random variables.
"""
from .randomprocess import *
from .kernels import *
from .gaussianprocess import *

__all__ = ["RandomProcess", "GaussianProcess",
           "Kernel", "IntegratedBrownianMotionKernel",
           "BrownianMotionKernel"]

# Set correct module paths (for superclasses). Corrects links and module paths in documentation.
RandomProcess.__module__ = "probnum.prob.randomprocess"
