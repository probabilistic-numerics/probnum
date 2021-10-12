"""Test cases defined by policies."""
from pytest_cases import case

from probnum.linalg.solvers import policies


def case_conjugate_gradient():
    return policies.ConjugateGradientPolicy()


@case(tags=["random"])
def case_random_unit_vector():
    return policies.RandomUnitVectorPolicy()
