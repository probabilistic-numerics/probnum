"""Continuous-time transition utility functions."""

import functools

import numpy as np
import scipy.linalg

import probnum.random_variables as pnrv


def matrix_fraction_decomposition(driftmat, dispmat, step):
    """Matrix fraction decomposition (assuming no force)."""

    topleft = driftmat
    topright = dispmat @ dispmat.T
    bottomright = -driftmat.T
    bottomleft = np.zeros(driftmat.shape)

    toprow = np.hstack((topleft, topright))
    bottomrow = np.hstack((bottomleft, bottomright))
    bigmat = np.vstack((toprow, bottomrow))

    Phi = scipy.linalg.expm(bigmat * step)
    projmat1 = np.eye(*toprow.shape)
    projmat2 = np.flip(projmat1)

    Ah = projmat1 @ Phi @ projmat1.T
    C, D = projmat1 @ Phi @ projmat2.T, Ah.T
    Qh = C @ D

    return Ah, Qh, bigmat
