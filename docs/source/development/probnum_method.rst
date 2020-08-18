Implementing a Probabilistic Numerical Method
=============================================

Probabilistic numerical methods in ProbNum follow the dogma of having random variables as input and random variables as
output. Hence their signature should be similar to

.. code-block:: python

	randvar_out = probnum_method(problem, randvar_in, **kwargs)


where :code:`problem` defines the numerical problem to solve, :code:`randvar_in` is the input random variable encoding
prior information and :code:`**kwargs` are keyword arguments that influence the behaviour
of the solver (e.g. control termination). If your method has a classic analogue in NumPy or SciPy, make sure the 
signatures match as closely as possible. This enables PN methods to be used as drop-in replacements for classic 
numerical routines.

Interface and Implementation
****************************

Each method in ProbNum has an interface called by the user defined through a function similar to the one below.

.. code-block:: python

	def my_probnum_method(problem, randvar_in, type="vanilla", **kwargs):
	    """
	    Solve a numerical problem using a probabilistic numerical method.

	    This probabilistic numerical method solves the problem xyz, by ...

	    Parameters
	    ----------
	    problem : Problem, shape=(n,n)
	        Arguments defining the numerical problem to be solved.
	    randvar_in : RandomVariable, shape=(n,)
	        Input random variable encoding prior information about the problem.
	    type : str
	        Variant of probabilistic numerical method to use. The available options are

	        ====================  ===========
	         vanilla              ``vanilla``
	         advanced              ``adv``
	        ====================  ===========

	    Returns
	    -------
	    randvar_out : RandomVariable, shape=(n,)
	        Output random variable with posterior distribution over the quantity to be estimated.

	    Raises
	    ------
	    ValueError
	        Input shapes do not match.
	    """
	    # Choose method
	    if type == "vanilla":
	        pnmethod = VanillaPnMethod(problem, randvar_in)
	    elif type == "wasabi":
	        pnmethod = AdvancedPnMethod(problem, randvar_in)

	    # Solve problem
	    randvar_out, info = pnmethod.solve(**kwargs)

	    # Return output with information (e.g. on convergence)
	    return randvar_out, info


This interface is separate from the actual implementation(s) of the PN method. Often there are different variations of a
given numerical routine depending on the arguments supplied. These are implemented in a class hierarchy usually in the same module
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
	    randvar_in : RandomVariable, shape=(n,)
	        Input random variable encoding prior information about the problem.
	    """

	    def __init__(self, problem: Problem, randvar_in: prob.RandomVariable):
	        raise NotImplementedError
	    
	    def solve(self, **kwargs) -> Tuple[prob.RandomVariable, dict]:
	        raise NotImplementedError


	class VanillaPnMethod(PnMethod):

	    def __init__(self, problem: Problem, randvar_in: prob.RandomVariable):
	        raise NotImplementedError

	    def solve(self) -> Tuple[prob.RandomVariable, dict]:
	        """
	        Solve the numerical problem in a basic way.
	        
	        Returns
	        -------
	        randvar_out : RandomVariable
	            Posterior distribution over the solution of `problem`.
	        info : dict
	            Information on the convergence of the iteration.
	        """
	        raise NotImplementedError


	class AdvancedPnMethod(PnMethod):

	    def __init__(self, problem: Problem, randvar_in: prob.RandomVariable):
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
	        randvar_out : RandomVariable
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