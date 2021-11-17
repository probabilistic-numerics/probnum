"""Test cases defining probabilistic linear solvers."""

from pytest_cases import case

from probnum.linalg import solvers


@case(tags=["solutionbased"])
def case_bayescg():
    return solvers.BayesCG()


@case(tags=["solutionbased"])
def case_probkaczmarz():
    return solvers.ProbabilisticKaczmarz()


@case(tags=["matrixbased"])
def case_matrixbasedpls():
    return solvers.MatrixBasedPLS()


@case(tags=["matrixbased"])
def case_symmatrixbasedpls():
    return solvers.SymMatrixBasedPLS()
