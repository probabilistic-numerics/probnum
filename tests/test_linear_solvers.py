import pytest
import numpy as np
import probnum.linalg.linear_solvers


@pytest.mark.parametrize("prob_linear_solver",
                         [probnum.linalg.linear_solvers.MatrixBasedConjugateGradients()])
def test_dimension_mismatch_exception(prob_linear_solver):
    """Test whether linear solvers throw an exception for input with mismatched dimensions."""
    A = np.zeros(shape=[3, 2])
    b = np.zeros(shape=[4])
    with pytest.raises(ValueError, match="Dimension mismatch.") as e:
        assert prob_linear_solver.solve(A=A, b=b), "Invalid input formats should raise a ValueError."
