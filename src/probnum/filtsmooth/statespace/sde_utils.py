"""Continuous-time transition utility functions."""

import functools

import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv


def matrix_fraction_decomposition(driftmat, dispmat, dt):
    """Matrix fraction decomposition (assuming no force)."""
    dim = len(driftmat)
    dispmat = np.squeeze(dispmat)
    Phi = np.block(
        [
            [driftmat, np.outer(dispmat, dispmat)],
            [np.zeros(driftmat.shape), -driftmat.T],
        ]
    )
    M = scipy.linalg.expm(Phi * dt)

    Ah = M[:dim, :dim]
    Qh = M[:dim, dim:] @ Ah.T

    return Ah, Qh, M
