=========
Tutorials
=========

Learn how to use ProbNum and get to know its features. You can interactively try out the |Tutorials| directly in the browser or
by downloading the notebooks from the
`GitHub repository <https://github.com/probabilistic-numerics/probnum/tree/main/docs/source/tutorials>`_.


Getting Started
***************

.. nbgallery::
   :caption: Getting Started

   tutorials/quickstart
   tutorials/pn_methods


Features of ProbNum
*******************

Linear Solvers
--------------

Solving linear systems is arguably one of the most fundamental computations in statistics, machine learning and numerics. For example, linear systems arise when inferring parameters in statistical models or during model training.
ProbNum provides a family of linear solvers, which infer either the inverse system matrix or the solution directly, while quantifying their uncertainty.


.. nbgallery::
   :caption: Linear Solvers

   tutorials/linalg/linear_systems
   tutorials/linalg/galerkin_method


Ordinary Differential Equation Solvers
--------------------------------------

The behaviour of complex systems which evolve over time is often described via the use of ordinary differential equations.
ProbNum provides a set of methods to solve ordinary differential equations based on filtering approaches which quantify
the uncertainty introduced by discretization.


.. nbgallery::
   :caption: Differential Equation Solvers

   tutorials/odes/adaptive_steps_odefilter
   tutorials/odes/uncertainties_odefilters
   tutorials/odes/odesolvers_from_scratch
   tutorials/odes/event_handling


Bayesian Filtering and Smoothing
--------------------------------

Bayesian filtering and smoothing provides a framework for efficient inference in state space models.
For non-linear state space components, ProbNum provides linearization techniques that enables
Gaussian filtering and smoothing in more complex dynamical systems.

.. nbgallery::
   :caption: Bayesian Filtering and Smoothing

   tutorials/filtsmooth/linear_gaussian_filtering_smoothing
   tutorials/filtsmooth/nonlinear_gaussian_filtering_smoothing
   tutorials/filtsmooth/particle_filtering


Linear Operators
----------------

Linear algebra is essential to virtually all of scientific computation.
ProbNum implements (finite-dimensional) linear operators in a memory-efficient manner with support for lazy arithmetic.

.. nbgallery::
   :caption: Linear Operators

   tutorials/linops/linear_operators


Probability
-----------

Random variables are the main objects in ProbNum. They represent uncertainty about a numerical quantity via their
distribution. A probabilistic numerical method takes random variables as inputs and also outputs random variables.

.. nbgallery::
   :caption: Probability

   tutorials/prob/random_variables_quickstart


.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Jupyter-579ACA.svg?style=flat-square&logo=Jupyter&logoColor=white
    :target: https://mybinder.org/v2/gh/probabilistic-numerics/probnum/main?filepath=docs%2Fsource%2Ftutorials
    :alt: ProbNum's Tutorials
