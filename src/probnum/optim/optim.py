"""
Convenience functions for
0th, 1st and 2nd order local minimisation
of functions with np.ndarray input and scalar output (!!!).

Functionalities
---------------
* Random Search with constant learning rate and fixed number of steps
* GD, Newton, damped Newton (levenberg marquardt)
    * with stopping criterions: absolute tolerance, relative tolerance, maxit
    * with learning rates: constant, backtracking linesearch
"""

import numpy as np

from probnum.optim import objective, linesearch, stoppingcriterion

from probnum.optim.deterministic import steepestdescent, randomsearch


def minimise_rs(objec, initval, tol, lrate, maxit):
    """
    Convenience function for 0th order minimisation: random search (RS).

    Note
    ----
    By default, we use the DiffOfFctValues(tol) stopping criterion. If you
    want constant number of steps, set tol=0.0.

    Parameters
    ----------
    objec: callable(), e.g. a lambda function
        Objective function to be minimised
    initval: np.ndarray or scalar
        Initial value for minimisation
    lrate: float
        Learning rate for RS. RS does not support line search methods
    nsteps: int
        Number of iterations.

    Returns
    -------
    traj: np.ndarray of shape (nsteps, len(initval))
        Full ptimisation trajectory
    obs: np.ndarray of shape (nsteps)
        Function values of objective at each element of traj

    Example
    --------
    >>> import numpy as np
    >>> np.random.seed(1)   # for successful reproduction

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([15.0])
    >>> traj, objs = minimise_rs(obj, initval, tol=0.01, lrate=0.65, maxit=150)
    >>> print(traj[-1])
    [0.05]
    """
    stopcrit = stoppingcriterion.DiffOfFctValues(tol)
    lsearcher = linesearch.ConstantLearningRate(lrate)
    adobj = objective.Objective(objec)
    rsminimiser = randomsearch.RandomSearch(lsearcher, stopcrit, maxit)
    traj, obs = rsminimiser.minimise_nd(adobj, initval)
    return traj, obs


def minimise_gd(objec, derivative, initval, tol,
                lrate=None, lsearch=None, maxit=1000):
    """
    Convenience function for 1st order minimisation: Gradient descent

    Note
    ----
    By default, we use the NormOfGradient(tol) stopping criterion. If you
    want constant number of steps, set tol=0.0.


    Examples
    --------

    Minimisation using constant learning rate

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([150.0])
    >>> traj, objs = minimise_gd(obj, der, initval, tol=1e-5, lrate=0.75)
    >>> print(traj[-1])
    [-4.47034836e-06]

    Minimisation using backtracking linesearch

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([150.0])
    >>> traj, objs = minimise_gd(obj, der, initval, tol=1e-1, lsearch='backtrack')
    >>> print(traj[-1])
    [0.01396984]
    """
    stopcrit = stoppingcriterion.NormOfGradient(tol)
    lsearcher = _create_lsearcher(lrate, lsearch)
    adobj = objective.Objective(objec, derivative)
    gdminimiser = steepestdescent.GradientDescent(lsearcher, stopcrit, maxit)
    return gdminimiser.minimise_nd(adobj, initval)


def minimise_newton(objec, derivative, hessian, initval, tol,
                    lrate=None, lsearch=None, maxit=1000):
    """
    Convenience function for 2nd order minimisation.

    Note
    ----
    By default, we use the NormOfGradient(tol) stopping criterion. If you
    want constant number of steps, set tol=0.0.

    Examples
    --------

    Minimisation using constant learning rate

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([150.0])
    >>> traj, objs = minimise_newton(obj, der, hess, initval, tol=1e-1, lrate=.75)
    >>> print(traj[-1])
    [0.03662109]

    Minimisation using backtracking linesearch

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([150.0])
    >>> traj, objs = minimise_newton(obj, der, hess, initval, \
                                     tol=1e-5, lsearch='backtrack')
    >>> print(traj[-1])
    [0.]
    """
    stopcrit = stoppingcriterion.NormOfGradient(tol)
    lsearcher = _create_lsearcher(lrate, lsearch)
    adobj = objective.Objective(objec, derivative, hessian)
    newtonminimiser = steepestdescent.NewtonMethod(lsearcher, stopcrit, maxit)
    return newtonminimiser.minimise_nd(adobj, initval)


def minimise_levmarq(objec, derivative, hessian, initval, dampingpar, tol,
                     lrate=None, lsearch=None, maxit=1000):
    """
    Convenience function for 2nd order minimisation.

    Note
    ----
    By default, we use the NormOfGradient(tol) stopping criterion. If you
    want constant number of steps, set tol=0.0.

    Examples
    --------

    Minimisation using constant learning rate

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([150.0])
    >>> traj, objs = minimise_levmarq(obj, der, hess, initval, \
                                      dampingpar=0.1, tol=1e-1, lrate=0.75)
    >>> print(traj[-1])
    [0.0233139]

    Minimisation using backtracking linesearch

    >>> def obj(x):
    ...     return x.T @ x
    ...
    >>> def der(x):
    ...     return 2.0*x
    ...
    >>> def hess(x):
    ...     return 2.0 * np.ones((len(x), len(x)))
    ...
    >>> initval = np.array([150.0])
    >>> traj, objs = minimise_levmarq(obj, der, hess, initval, dampingpar=0.1, \
                                      tol=1e-5, lsearch='backtrack')
    >>> print(traj[-1])
    [1.74894234e-06]
    """
    stopcrit = stoppingcriterion.NormOfGradient(tol)
    lsearcher = _create_lsearcher(lrate, lsearch)
    adobj = objective.Objective(objec, derivative, hessian)
    levmarqmini = steepestdescent.LevenbergMarquardt(dampingpar, lsearcher,
                                                     stopcrit, maxit)
    return levmarqmini.minimise_nd(adobj, initval)


def _create_lsearcher(lrate, lsearch):
    """
    """
    if lrate is None and lsearch is None:
        raise ValueError("Please enter e.g. lrate=0.1 or lsearch='backtrack'")
    if lrate is None:
        assert lsearch == "backtrack", "Please enter lsearch='backtrack'"
        lsearcher = linesearch.BacktrackingLineSearch()
    elif lsearch is None:
        assert lrate > 0, "Please enter a positive learning rate"
        lsearcher = linesearch.ConstantLearningRate(lrate)
    else:
        raise ValueError("Please enter either lrate or lsearch, not both")
    return lsearcher
