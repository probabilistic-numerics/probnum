"""Test cases describing different belief updates over quantities of interest of a
linear system."""
from pytest_cases import parametrize

from probnum.linalg.solvers import belief_updates


@parametrize(noise_var=[0.0, 1.0])
def case_solution_based_projected_rhs_belief_update(noise_var: float):
    return belief_updates.SolutionBasedProjectedRHSBeliefUpdate(noise_var=noise_var)


def case_symmetric_matrix_based_linear_belief_update():
    return belief_updates.SymmetricMatrixBasedLinearBeliefUpdate()
