probnum.filtsmooth
==================

Bayesian Filtering and Smoothing.

This package provides different kinds of Bayesian filters and smoothers which
estimate the distribution over observed and hidden variables in a sequential model.
The two operations differ by what information they use. Filtering considers all
observations up to a given point, while smoothing takes the entire set of
observations into account.

.. automodapi:: probnum.filtsmooth
    :no-heading:
    :no-main-docstr:


probnum.filtsmooth.statespace
=============================

Probabilistic State Space Models.

This package implements continuous-discrete and discrete-discrete state space models,
which are the basis for Bayesian filtering and smoothing, but also probabilistic ODE solvers.

.. automodapi:: probnum.filtsmooth.statespace
    :no-heading:
    :no-main-docstr:
