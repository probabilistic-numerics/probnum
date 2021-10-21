"""Test cases defined by information operators."""

from probnum.linalg.solvers import information_ops


def case_matvec():
    return information_ops.MatVecInformationOp()


def case_projected_residual():
    return information_ops.ProjectedResidualInformationOp()
