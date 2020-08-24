from ._random_variable import RandomVariable, asrandvar

from ._dirac import Dirac
from ._normal import Normal

# Set correct module paths. Corrects links and module paths in documentation.
asrandvar.__module__ = "probnum.random_variables"

RandomVariable.__module__ = "probnum.random_variables"
Dirac.__module__ = "probnum.random_variables"
Normal.__module__ = "probnum.random_variables"
