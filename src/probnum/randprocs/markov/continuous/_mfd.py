"""Continuous-time transition utility functions."""


import numpy as np
import scipy.linalg


def matrix_fraction_decomposition(drift_matrix, dispersion_matrix, dt):
    """Matrix fraction decomposition (assuming no force)."""
    dim = drift_matrix.shape[0]

    if dispersion_matrix.ndim == 1:
        dispersion_matrix = dispersion_matrix.reshape((-1, 1))

    Phi = np.block(
        [
            [drift_matrix, dispersion_matrix @ dispersion_matrix.T],
            [np.zeros(drift_matrix.shape), -drift_matrix.T],
        ]
    )
    M = scipy.linalg.expm(Phi * dt)

    Ah = M[:dim, :dim]
    Qh = M[:dim, dim:] @ Ah.T

    return Ah, Qh, M
