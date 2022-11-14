"""Test cases defining probabilistic linear solvers."""

from probnum.linalg import solvers

from pytest_cases import case


@case(tags=["solutionbased", "sym"])
def case_bayescg():
    return solvers.BayesCG()


@case(tags=["solutionbased"])
def case_probkaczmarz():
    return solvers.ProbabilisticKaczmarz()


@case(tags=["matrixbased"])
def case_matrixbasedpls():
    return solvers.MatrixBasedPLS()


@case(tags=["matrixbased", "sym"])
def case_symmatrixbasedpls():
    return solvers.SymMatrixBasedPLS()
