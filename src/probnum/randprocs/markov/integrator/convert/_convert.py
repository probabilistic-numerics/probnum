"""Conversion functions for representations of integrators."""

import numpy as np

from probnum.randprocs.markov.integrator import _integrator
from probnum.typing import IntArgType


def convert_derivwise_to_coordwise(
    state: np.ndarray, num_derivatives: IntArgType, wiener_process_dimension: IntArgType
) -> np.ndarray:
    """Convert coordinate-wise representation to derivative-wise representation.

    Lightweight call to the respective property in :class:`Integrator`.

    Parameters
    ----------
    state:
        State to be converted. Assumed to be in coordinate-wise representation.
    num_derivatives:
        Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
    wiener_process_dimension:
        Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.
    """
    projmat = _integrator.IntegratorTransition(
        num_derivatives, wiener_process_dimension
    )._derivwise2coordwise_projmat
    return projmat @ state


def convert_coordwise_to_derivwise(
    state: np.ndarray, num_derivatives: IntArgType, wiener_process_dimension: IntArgType
) -> np.ndarray:
    """Convert coordinate-wise representation to derivative-wise representation.

    Lightweight call to the respective property in :class:`Integrator`.

    Parameters
    ----------
    state:
        State to be converted. Assumed to be in derivative-wise representation.
    num_derivatives:
        Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
    wiener_process_dimension:
        Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.
    """
    projmat = _integrator.IntegratorTransition(
        num_derivatives, wiener_process_dimension
    )._coordwise2derivwise_projmat
    return projmat @ state
