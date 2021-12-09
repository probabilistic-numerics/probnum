"""Test cases defined by policies."""
from pytest_cases import case

from probnum.linalg.solvers import policies
from probnum.utils.linalg import modified_gram_schmidt


def case_conjugate_gradient():
    return policies.ConjugateGradientPolicy()


def case_conjugate_gradient_reorthogonalized():
    return policies.ConjugateGradientPolicy(
        reorthogonalization_fn=modified_gram_schmidt
    )


@case(tags=["random"])
def case_random_unit_vector():
    return policies.RandomUnitVectorPolicy()
