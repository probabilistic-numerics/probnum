"""Conversion functions for representations of integrators."""

import numpy as np

from probnum.randprocs.markov.continuous.integrator import _integrator
from probnum.typing import IntArgType


def convert_derivwise_to_coordwise(
    state: np.ndarray, ordint: IntArgType, spatialdim: IntArgType
) -> np.ndarray:
    """Convert coordinate-wise representation to derivative-wise representation.

    Lightweight call to the respective property in :class:`Integrator`.

    Parameters
    ----------
    state:
        State to be converted. Assumed to be in coordinate-wise representation.
    ordint:
        Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
    spatialdim:
        Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.
    """
    projmat = _integrator.IntegratorTransition(
        ordint, spatialdim
    )._derivwise2coordwise_projmat
    return projmat @ state


def convert_coordwise_to_derivwise(
    state: np.ndarray, ordint: IntArgType, spatialdim: IntArgType
) -> np.ndarray:
    """Convert coordinate-wise representation to derivative-wise representation.

    Lightweight call to the respective property in :class:`Integrator`.

    Parameters
    ----------
    state:
        State to be converted. Assumed to be in derivative-wise representation.
    ordint:
        Order of the integrator-state. Usually, this is the order of the highest derivative in the state.
    spatialdim:
        Spatial dimension of the integrator. Usually, this is the number of states associated with each derivative.
    """
    projmat = _integrator.IntegratorTransition(
        ordint, spatialdim
    )._coordwise2derivwise_projmat
    return projmat @ state
