Implementing a Probabilistic Numerical Method
=============================================

Probabilistic numerical methods in ProbNum follow the dogma of having random variables as input and random variables as
output. Hence their signature should be similar to

.. code-block:: python

	rv_out = probnum_method(problem, rv_in, **kwargs)


where :code:`problem` defines the numerical problem to solve, :code:`rv_in` is the input random variable encoding
prior information and :code:`**kwargs` are keyword arguments that influence the behaviour of the solver (e.g. control
termination).

Example Implementation of a PN Method
**************************************

Consider the problem of finding the scalar root :math:`x` of an equation :math:`f(x)=0`. Suppose we have developed an
iterative PN method for this problem, which computes a belief over the location of the root. In each iteration :math:`i`
we evaluate :math:`f` at a location :math:`x_i` to update our belief about the solution :math:`x`. This choice of where
to evaluate is given by the *policy* of the algorithm. Once the algorithm determines (e.g. based on its belief) that the
estimated root is close to the true root, it terminates. We will now mock-up this hypothetical PN method to demonstrate
how PN methods are implemented in ProbNum.

Interface
"""""""""
Each method in ProbNum has a user interface defined by a function similar to the one below. If your method has a classic
analogue in NumPy or SciPy, make sure the signatures match as closely as possible. This enables PN methods to be used as
drop-in replacements for classic numerical routines.

.. code-block:: python

    from typing import Callable, Dict, Optional

    from probnum import RandomVariable


    def bayes_root_scalar(
        f: Callable[[float], float],
        rv_in: RandomVariable,
        method: str = "vanilla",
        maxiter: Optional[int] = None,
        atol: float = 10 ** -6,
        rtol: float = 10 ** -6,
    ) -> Tuple[RandomVariable, Dict]:
        """
        Find a root of a scalar function.

        Iteratively computes a probabilistic belief over the root of ``f``.

        Parameters
        ----------
        f :
            Argument(s) defining the numerical problem to be solved. These could be a function, a matrix, vectors, etc.
        rv_in :
            Input random variable encoding prior information about the problem.
        method :
            Variant of probabilistic numerical method to use. The available options are

            ====================  ===========
             vanilla              ``vanilla``
             advanced              ``adv``
            ====================  ===========

        maxiter :
            Maximum number of iterations.
        atol :
            Absolute convergence tolerance.
        rtol :
            Relative convergence tolerance.

        Returns
        -------
        rv_out :
            Output random variable with posterior distribution over the quantity to be estimated.

        Raises
        ------
        ValueError
            Input shapes do not match.
        """
        # Instantiate a variant of the PN method based on the problem and arguments
        if type == "vanilla":
            pnmethod = PnMethod(
                problem=problem,
                rv_in=rv_in,
                policy=deterministic_policy,
                stopping_criteria=[residual_stopping_criterion,],
            )
        elif type == "adv":
            pnmethod = PnMethod(
                problem=problem,
                rv_in=rv_in,
                policy=max_uncertainty_policy,
                stopping_criteria=probabilistic_stopping_criterion,
            )

        # Solve problem
        rv_out, info = pnmethod.solve(**kwargs)

        # Return output with information (e.g. on convergence)
        return rv_out, info


This interface is separate from the actual implementation(s) of the PN method.

Implementation
""""""""""""""


Often there are different variations of a given numerical routine depending on the arguments supplied. These are
implemented in a class hierarchy usually in the same module
as the interface. In order to decrease pesky type bugs and increase maintainability these implementations must have `type
hints <https://docs.python.org/3/library/typing.html>`_.

.. code-block:: python

	from typing import Tuple
	from probnum import prob

	class PnMethod:
	    """
	    Probabilistic numerical method

	    Parameters
	    ----------
	    problem : Problem, shape=(n,n)
	        Arguments defining the numerical problem to be solved.
	    rv_in : RandomVariable, shape=(n,)
	        Input random variable encoding prior information about the problem.
	    """

	    def __init__(self, problem: Problem, rv_in: prob.RandomVariable):
	        raise NotImplementedError
	    
	    def solve(self, **kwargs) -> Tuple[prob.RandomVariable, dict]:
	        raise NotImplementedError


	class VanillaPnMethod(PnMethod):

	    def __init__(self, problem: Problem, rv_in: prob.RandomVariable):
	        raise NotImplementedError

	    def solve(self) -> Tuple[prob.RandomVariable, dict]:
	        """
	        Solve the numerical problem in a basic way.
	        
	        Returns
	        -------
	        rv_out : RandomVariable
	            Posterior distribution over the solution of `problem`.
	        info : dict
	            Information on the convergence of the iteration.
	        """
	        raise NotImplementedError


	class AdvancedPnMethod(PnMethod):

	    def __init__(self, problem: Problem, rv_in: prob.RandomVariable):
	        raise NotImplementedError

	    def solve(self, maxiter: int) -> Tuple[prob.RandomVariable, dict]:
	        """
	        Solve the numerical problem in an advanced way.
	        
	        Parameters
	        ----------
	        maxiter : int
	            Maximum number of iterations of the solve loop.

	        Returns
	        -------
	        rv_out : RandomVariable
	            Posterior distribution over the solution of `problem`.
	        info : dict
	            Information on the convergence of the iteration.
	        """
	        raise NotImplementedError


Before you add a new method interface or class, look through the codebase whether you can simply subclass an existing 
implementation of a PN method.

Testing
********

While or even before you add a new PN method, write tests for its functionality. Writing tests before the 
code forces you to think about what your numerical method should do independent of its implementation. Some basic tests
to consider are listed below.

In- and Output
"""""""""""""""
- **Deterministic input**: Does your method accept parameters / problem definitions which are not random variables?
- **Shape**: Does your PN method return consistent shapes for differently shaped inputs?
- **Expected errors**: Are appropriate errors raised for invalid input?

Numerical
""""""""""
- **Perfect information**: Does your method converge instantly for a prior encoding the solution of the problem?
- **Convergence criteria**: Are all convergence criteria covered by at least one test?

Conclusion
***********
Once you are done, consider writing an `example notebook <https://probnum.readthedocs.io/tutorials/tutorials.html>`_
showcasing your new method. Congratulations you just implemented your first probabilistic numerical method in ProbNum!