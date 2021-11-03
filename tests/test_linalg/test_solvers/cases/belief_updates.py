"""Test cases describing different belief updates over quantities of interest of a
linear system."""
from pytest_cases import parametrize

from probnum.linalg.solvers.belief_updates import matrix_based, solution_based


@parametrize(noise_var=[0.0, 0.001, 1.0])
def case_solution_based_projected_rhs_belief_update(noise_var: float):
    return solution_based.SolutionBasedProjectedRHSBeliefUpdate(noise_var=noise_var)


def case_matrix_based_linear_belief_update():
    return matrix_based.MatrixBasedLinearBeliefUpdate()


def case_symmetric_matrix_based_linear_belief_update():
    return matrix_based.SymmetricMatrixBasedLinearBeliefUpdate()
