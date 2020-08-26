"""
This package implements random variables. Random variables are the primary in- and
outputs of probabilistic numerical methods. A generic signature of such methods looks
like this:

.. highlight:: python
.. code-block:: python

    randvar_out, info = probnum_method(problem, randvar_in, **kwargs)

"""

from ._random_variable import (
    asrandvar,
    RandomVariable,
    DiscreteRandomVariable,
    ContinuousRandomVariable,
)

from ._dirac import Dirac
from ._normal import Normal
