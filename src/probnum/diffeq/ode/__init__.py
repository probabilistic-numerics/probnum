from .ivp import IVP
from .ivp_examples import (  # rigidbody,; threebody,; vanderpol,
    fitzhughnagumo,
    logistic,
    lorenz,
    lotkavolterra,
    seir,
)
from .ode import ODE

__all__ = [
    "ODE",
    "IVP",
    "logistic",
    "fitzhughnagumo",
    "seir",
    # "rigidbody",
    # "vanderpol",
    # "threebody",
    "lotkavolterra",
    "lorenz",
]
