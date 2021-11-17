"""Test cases describing different beliefs over quantities of interest of a linear
system."""

import numpy as np
from pytest_cases import case

from probnum import linops, randvars
from probnum.linalg.solvers import beliefs


def case_trivial_prior(ncols: int) -> beliefs.LinearSystemBelief:
    return beliefs.LinearSystemBelief(
        x=randvars.Normal(mean=np.zeros((ncols,)), cov=linops.Identity(ncols)),
        Ainv=randvars.Normal(
            mean=np.zeros((ncols, ncols)),
            cov=linops.SymmetricKronecker(linops.Identity(ncols)),
        ),
        A=randvars.Normal(
            mean=np.zeros((ncols, ncols)),
            cov=linops.SymmetricKronecker(linops.Identity(ncols)),
        ),
    )
