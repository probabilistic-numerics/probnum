"""Utilities."""
from probnum import randvars

# public (because it is needed in some integrator implementations),
# but not exposed to the 'randprocs' namespace
# (i.e. not imported in any __init__.py).
__all__ = ["apply_precon"]


def apply_precon(precon, rv):

    # There is no way of checking whether `rv` has its Cholesky factor computed already or not.
    # Therefore, since we need to update the Cholesky factor for square-root filtering,
    # we also update the Cholesky factor for non-square-root algorithms here,
    # which implies additional cost.
    # See Issues #319 and #329.
    # When they are resolved, this function here will hopefully be superfluous.

    new_mean = precon @ rv.mean
    new_cov_cholesky = precon @ rv.cov_cholesky  # precon is diagonal, so this is valid
    new_cov = new_cov_cholesky @ new_cov_cholesky.T

    return randvars.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky)
