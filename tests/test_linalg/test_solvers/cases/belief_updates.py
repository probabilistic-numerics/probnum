"""Test cases describing different belief updates over quantities of interest of a
linear system."""
from probnum.linalg.solvers import belief_updates


def case_solution_based_projected_residual_belief_update():
    return belief_updates.SolutionBasedProjectedResidualBeliefUpdate()


def case_symmetric_matrix_based_linear_belief_update():
    return belief_updates.SymmetricMatrixBasedLinearBeliefUpdate()
