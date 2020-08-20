from ._dirac import Dirac
from ._normal import Normal

# Set correct module paths. Corrects links and module paths in documentation.
Dirac.__module__ = "probnum.prob.random_variable"
Normal.__module__ = "probnum.prob.random_variable"
