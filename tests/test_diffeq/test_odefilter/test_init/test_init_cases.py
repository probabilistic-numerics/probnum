"""Test cases for initialization."""

import pytest_cases
from jax.config import config  # speed...

from probnum.diffeq.odefilter import init
from probnum.problems.zoo import diffeq as diffeq_zoo

from . import known_initial_derivatives

config.update("jax_disable_jit", True)


@pytest_cases.case(tags=("jax",))
def problem_threebody():
    ivp = diffeq_zoo.threebody_jax()
    threebody_inits_matrix_full = known_initial_derivatives.THREEBODY_INITS
    return ivp, threebody_inits_matrix_full


@pytest_cases.case(tags=("numpy",))
def problem_lotka_volterra():
    ivp = diffeq_zoo.lotkavolterra()
    inits_matrix_full = known_initial_derivatives.LV_INITS
    return ivp, inits_matrix_full


@pytest_cases.case(tags=["is_exact", "requires_jax"])
def solver_taylor_mode():
    return init.TaylorMode()


@pytest_cases.case(tags=["is_exact", "requires_jax"])
def solver_auto_diff_forward_jvp():
    return init.ForwardModeJVP()


@pytest_cases.case(tags=["is_exact", "requires_jax"])
def solver_auto_diff_forward():
    return init.ForwardMode()


@pytest_cases.case(tags=["is_exact", "requires_jax"])
def solver_auto_diff_reverse():
    return init.ReverseMode()


@pytest_cases.case(tags=["is_not_exact", "requires_numpy"])
def solver_runge_kutta():
    return init.RungeKutta()


#
# @pytest_cases.case(tags=["is_not_exact", "requires_numpy"])
# def solver_runge_kutta_with_jacobian():
#     return init.RungeKuttaWithJacobian()


@pytest_cases.case(tags=["is_not_exact", "requires_numpy"])
def solver_stack():
    return init.Stack()


@pytest_cases.case(tags=["is_not_exact", "requires_numpy"])
def solver_stack_with_jacobian():
    return init.StackWithJacobian()
