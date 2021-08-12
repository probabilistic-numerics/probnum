"""Conversion functions for representations of integrators."""

import numpy as np

from probnum.randprocs.markov.integrator import _iwp
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
    dummy_integrator = _iwp.IntegratedWienerTransition(
        num_derivatives=num_derivatives,
        wiener_process_dimension=wiener_process_dimension,
    )

    return dummy_integrator.reorder_state(
        state, current_ordering="derivative", target_ordering="coordinate"
    )


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
    dummy_integrator = _iwp.IntegratedWienerTransition(
        num_derivatives=num_derivatives,
        wiener_process_dimension=wiener_process_dimension,
    )

    return dummy_integrator.reorder_state(
        state, current_ordering="coordinate", target_ordering="derivative"
    )
