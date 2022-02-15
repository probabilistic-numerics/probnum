"""Test cases defined by policies."""
from pytest_cases import case

from probnum.linalg.solvers import policies
from probnum.utils.linalg import double_gram_schmidt, modified_gram_schmidt


def case_conjugate_gradient():
    return policies.ConjugateGradientPolicy()


def case_conjugate_gradient_reorthogonalized_residuals():
    return policies.ConjugateGradientPolicy(
        reorthogonalization_fn_residual=double_gram_schmidt
    )


def case_conjugate_gradient_reorthogonalized_actions():
    return policies.ConjugateGradientPolicy(
        reorthogonalization_fn_action=modified_gram_schmidt
    )


@case(tags=["random"])
def case_random_unit_vector():
    return policies.RandomUnitVectorPolicy()


@case(tags=["random"])
def case_random_unit_vector_no_replacement():
    return policies.RandomUnitVectorPolicy(replace=False)


@case(tags=["random"])
def case_random_unit_vector_rownorm_probs():
    return policies.RandomUnitVectorPolicy(probabilities="rownorm")
