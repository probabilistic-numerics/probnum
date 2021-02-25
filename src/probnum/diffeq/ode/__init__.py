from .ivp import IVP
from .ivp_examples import fitzhughnagumo, logistic, lorenz, lotkavolterra, seir
from .ode import ODE

__all__ = [
    "ODE",
    "IVP",
    "logistic",
    "fitzhughnagumo",
    "seir",
    "lotkavolterra",
    "lorenz",
]
