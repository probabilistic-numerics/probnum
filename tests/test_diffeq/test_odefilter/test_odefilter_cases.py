import numpy as np
import pytest
import pytest_cases

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, randprocs


# logistic.rhs is implemented backend-agnostic,
# thus it works for both numpy and jax
@pytest_cases.case(tags=("numpy", "jax"))
def problem_logistic():
    return diffeq_zoo.logistic()


def steprule_constant():
    return diffeq.stepsize.ConstantSteps(0.5)


def steprule_adaptive():
    return diffeq.stepsize.AdaptiveSteps(firststep=0.5, atol=0.2, rtol=0.2)


def diffusion_constant():
    return randprocs.markov.continuous.ConstantDiffusion()


def diffusion_piecewise_constant():
    return randprocs.markov.continuous.PiecewiseConstantDiffusion(t0=0.0)


@pytest_cases.case(tags=("numpy",))
def init_scipy_fit():
    return diffeq.odefilter.init.SciPyFit()


@pytest_cases.case(tags=("numpy",))
def init_scipy_fit_with_jacobian():
    return diffeq.odefilter.init.SciPyFitWithJacobian()


@pytest_cases.case(tags=("numpy",))
def init_odefilter_map():
    return diffeq.odefilter.init.ODEFilterMAP()


@pytest_cases.case(tags=("numpy",))
def init_stack():
    return diffeq.odefilter.init.Stack()


@pytest_cases.case(tags=("numpy",))
def init_stack_with_jacobian():
    return diffeq.odefilter.init.StackWithJacobian()


@pytest_cases.case(tags=("jax",))
def init_forward():
    return diffeq.odefilter.init.ForwardMode()


@pytest_cases.case(tags=("jax",))
def init_forward_jvp():
    return diffeq.odefilter.init.ForwardModeJVP()


@pytest_cases.case(tags=("jax",))
def init_reverse():
    return diffeq.odefilter.init.ReverseMode()


@pytest_cases.case(tags=("jax",))
def init_taylor():
    return diffeq.odefilter.init.TaylorMode()


def approx_ek0():
    return diffeq.odefilter.approx_strategies.EK0()


def approx_ek1():
    return diffeq.odefilter.approx_strategies.EK1()
