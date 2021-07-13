"""Propose a first step used for the ODE solver."""

import numpy as np


def propose_firststep(ivp):
    """Propose a suitable first step-size that can be taken by an ODE solver.

    This function implements a lazy version of the algorithm on p. 169
    of Hairer, Wanner, Norsett.
    """
    norm_y0 = np.linalg.norm(ivp.y0)
    norm_dy0 = np.linalg.norm(ivp.f(ivp.t0, ivp.y0))
    return 0.01 * norm_y0 / norm_dy0
