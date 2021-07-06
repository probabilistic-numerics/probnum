"""Test cases describing different beliefs over quantities of interest of a linear
system."""


from pytest_cases.case_parametrizer_new import parametrize_with_cases

from ...cases.linear_systems import case_linsys, case_spd_linsys


@parametrize_with_cases("linsys", cases=case_linsys)
def case_prior(linsys):
    """Prior linear system belief."""
    pass


@parametrize_with_cases("spd_linsys", cases=case_spd_linsys)
def case_perfect_spd_prior(spd_linsys):
    """Linear system belief concentrated at the true solution of a system with symmetric
    positive definite system matrix."""
    pass
