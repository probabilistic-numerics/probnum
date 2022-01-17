import numpy as np
import pytest
import pytest_cases

import probnum.problems.zoo.diffeq as diffeq_zoo
from probnum import diffeq, randprocs


def problem_logistic():
    return diffeq_zoo.logistic()


def steprule_constant():
    return diffeq.stepsize.ConstantSteps(0.1)


def steprule_adaptive():
    return diffeq.stepsize.AdaptiveSteps(firststep=0.1, atol=0.1, rtol=0.1)


def diffusion_constant():
    return randprocs.markov.continuous.ConstantDiffusion()


def diffusion_piecewise_constant():
    return randprocs.markov.continuous.PiecewiseConstantDiffusion(t0=0.0)


def init_scipy_fit():
    return diffeq.odefilter.init.SciPyFit()


def init_scipy_fit_with_jacobian():
    return diffeq.odefilter.init.SciPyFitWithJacobian()


def init_odefilter_map():
    return diffeq.odefilter.init.ODEFilterMAP()


def init_stack():
    return diffeq.odefilter.init.Stack()


def init_stack_with_jacobian():
    return diffeq.odefilter.init.StackWithJacobian()


def approx_ek0():
    return diffeq.odefilter.approx_strategies.EK0()


def approx_ek1():
    return diffeq.odefilter.approx_strategies.EK1()
