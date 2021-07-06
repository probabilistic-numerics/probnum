"""Test cases describing different beliefs over quantities of interest of a linear
system."""

import numpy as np
from pytest_cases.case_parametrizer_new import parametrize_with_cases

from probnum import linops, randvars
from probnum.linalg.solvers.beliefs import LinearSystemBelief

from ...cases.linear_systems import case_linsys, case_spd_linsys


@parametrize_with_cases("linsys", cases=case_linsys)
def case_prior(linsys):
    """Prior linear system belief."""
    return LinearSystemBelief(
        A=randvars.Constant(linsys.A),
        x=randvars.Normal(
            mean=np.zeros(linsys.A.shape[1]), cov=linops.Identity(shape=linsys.A.shape)
        ),
        b=randvars.Constant(linsys.b),
    )


@parametrize_with_cases("spd_linsys", cases=case_spd_linsys)
def case_spd_prior(spd_linsys):
    """Prior belief about a system with symmetric positive definite system matrix."""
    pass
