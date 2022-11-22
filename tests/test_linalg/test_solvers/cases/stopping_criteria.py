"""Stopping criteria test cases."""

from probnum.linalg.solvers import stopping_criteria

from pytest_cases import parametrize


def case_maxiter():
    return stopping_criteria.MaxIterationsStoppingCriterion()


def case_residual_norm():
    return stopping_criteria.ResidualNormStoppingCriterion()


@parametrize("qoi", ["x", "Ainv", "A"])
def case_posterior_contraction(qoi: str):
    return stopping_criteria.PosteriorContractionStoppingCriterion(qoi=qoi)
