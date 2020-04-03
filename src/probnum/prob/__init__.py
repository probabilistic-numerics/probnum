"""
Probability theory.

This package implements functionality related to probability theory such as random variables and distributions.
Random variables are the primary in- and outputs of probabilistic numerical methods. A generic signature of such methods
looks like this:

.. highlight:: python
.. code-block:: python

    randvar_out = probnum_method(randvar_in, **kwargs)

Examples
--------
>>> import numpy as np
>>> from probnum.prob import RandomVariable, Normal
>>>
>>> # Random seed
>>> np.random.seed(42)
>>> # Gaussian random variable
>>> X = RandomVariable(distribution=Normal(mean=0, cov=1))
>>> # Arithmetic operations between scalars and random variables
>>> Y = 2 * X - 3
>>> print(Y)
<() RandomVariable with dtype=<class 'float'>>

"""

from probnum.prob.interface import *
from probnum.prob.randomvariable import *
from probnum.prob.distributions import *

# Public classes and functions. Order is reflected in documentation.
__all__ = ["RandomVariable", "Distribution", "Dirac", "Normal", "asrandvar", "asdist"]
