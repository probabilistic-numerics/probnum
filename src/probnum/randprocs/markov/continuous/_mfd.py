"""Continuous-time transition utility functions."""


import numpy as np
import scipy.linalg


def matrix_fraction_decomposition(driftmat, dispmat, dt):
    """Matrix fraction decomposition (assuming no force)."""
    dim = driftmat.shape[0]

    if dispmat.ndim == 1:
        dispmat = dispmat.reshape((-1, 1))

    Phi = np.block(
        [
            [driftmat, dispmat @ dispmat.T],
            [np.zeros(driftmat.shape), -driftmat.T],
        ]
    )
    M = scipy.linalg.expm(Phi * dt)

    Ah = M[:dim, :dim]
    Qh = M[:dim, dim:] @ Ah.T

    return Ah, Qh, M
