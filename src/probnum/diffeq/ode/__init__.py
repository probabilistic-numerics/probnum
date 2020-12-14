from .ivp import IVP
from .ivp_examples import (
    fitzhughnagumo,
    logistic,
    lotkavolterra,
    rigidbody,
    seir,
    threebody,
    vanderpol,
)
from .ode import ODE

__all__ = [
    "ODE",
    "IVP",
    "logistic",
    "fitzhughnagumo",
    "seir",
    "rigidbody",
    "vanderpol",
    "threebody",
    "lotkavolterra",
]
