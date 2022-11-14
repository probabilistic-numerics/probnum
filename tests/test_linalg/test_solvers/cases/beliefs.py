"""Test cases describing different beliefs over quantities of interest of a linear
system."""

import numpy as np

from probnum import linops, randvars
from probnum.linalg.solvers import beliefs

from pytest_cases import case


@case(tags=["sym", "posdef", "square"])
def case_trivial_sym_prior(ncols: int) -> beliefs.LinearSystemBelief:
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


@case(tags=["square"])
def case_trivial_square_prior(ncols: int) -> beliefs.LinearSystemBelief:
    return beliefs.LinearSystemBelief(
        x=randvars.Normal(mean=np.zeros((ncols,)), cov=linops.Identity(ncols)),
        Ainv=randvars.Normal(
            mean=np.zeros((ncols, ncols)),
            cov=linops.Kronecker(linops.Identity(ncols), linops.Identity(ncols)),
        ),
        A=randvars.Normal(
            mean=np.zeros((ncols, ncols)),
            cov=linops.Kronecker(linops.Identity(ncols), linops.Identity(ncols)),
        ),
    )
