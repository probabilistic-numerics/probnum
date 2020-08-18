Adding to the Documentation
============================

Documentation is an integral part of every collaborative software project. Good documentation not only encourages users
of the package to try out different functionalities, but it also makes maintaining and expanding code significantly
easier. If you want to improve ProbNum's documentation or learn how to write documentation for your newly implemented
functionality, keep reading.

Docstrings
***********

The main form of documentation are `docstrings`, multi-line comments beneath a class or function definition with a
specific syntax, which detail its functionality. This package uses the
`NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide>`_. As a rule,
all functions which are exposed to the user *must* have appropriate docstrings. Below is an example of a docstring for a
probabilistic numerical method.

.. code-block:: python

	def problinsolve(A, b, A0=None, Ainv0=None, x0=None, assume_A="sympos", maxiter=None, atol=10 ** -6, rtol=10 ** -6,
	                 callback=None, **kwargs):
	    """
	    Infer a solution to the linear system :math:`A x = b` in a Bayesian framework.

	    Probabilistic linear solvers infer solutions to problems of the form

	    ...

	    Parameters
	    ----------
	    A : array_like or LinearOperator, shape=(n,n)
	        A square matrix or linear operator.

	    ...

	    Returns
	    -------
	    x : RandomVariable, shape=(n,) or (n, nrhs)
	        Approximate solution :math:`x` to the linear system. Shape of the return matches the shape of ``b``.

			...

	    Raises
	    ------
	    ValueError
	        If size mismatches detected or input matrices are not square.

	    Notes
	    -----
	    For a specific class of priors the probabilistic linear solver recovers the iterates of the conjugate gradient

	    ...

	    References
	    ----------
	    .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning, 2020

	    ...

	    See Also
	    --------
	    bayescg : Solve linear systems with prior information on the solution.

	    Examples
	    --------
	    >>> import numpy as np
	    >>> np.random.seed(1)
	    >>> n = 20
	    >>> A = np.random.rand(n, n)
	    >>> A = 0.5 * (A + A.T) + 5 * np.eye(n)
	    >>> b = np.random.rand(n)
	    >>> x, A, Ainv, info = problinsolve(A=A, b=b)
	    >>> print(info["iter"])
	    10
	    """


**General Rules**

- Cover :code:`Parameters`, :code:`Returns`, :code:`Raises` and :code:`Examples` at every publicly visible docstring---in that order.
- Examples are tested via doctest. Ensure :code:`doctest` does not fail.
- Include appropriate references, in particular for new probabilistic numerical methods.
- Do not use docstrings as a clutch for spaghetti code.

**Parameters**

- Hyperparameters should have default values and explanations on how to choose them.
- Array-type inputs (:code:`np.ndarray`, :code:`csr_matrix`, ...) are documented as :code:`array_like` (in accordance with NumPy).
- Provide shape hints via :code:`foo : array_like, shape=(a, b)` wherever possible.
- Callables are used as parameters if the expected signature is part of the docstring: :code:`foobar : callable, signature=(t, \*\*kwargs)`. Backslashes remove semantic meaning from special characters.

**Style**

- Stick to the imperative style of writing in the docstring header (i.e.: first line).

  - Yes: "Compute the value". 
  - No: "This function computes the value / Let's compute the value".
  
  The rest of the explanation talks about the function, e. g. "This function computes the value by computing another value".
- Use full sentences inside docstrings when describing something.

  - Yes: "This value is irrelevant, because it is not being passed on"
  - No: "Value irrelevant, not passed on". 
- When in doubt, more explanation rather than less. A little text inside an example can be helpful, too.
- A little maths can go a long way, but too much usually adds confusion.

Interface Documentation
************************

Which functions and classes actually show up in the documentation is determined by an :code:`__all__` statement in the 
corresponding :code:`__init__.py` file inside a module. The order of this list is also reflected in the documentation. 
For example, :code:`linalg` has the following :code:`__init__.py`:

.. code-block:: python

	"""
	Linear Algebra.

	This package implements common operations and (probabilistic) numerical methods for linear algebra.
	"""

	from probnum.linalg.linearsolvers import (
		problinsolve,
		bayescg,
		ProbabilisticLinearSolver,
	    MatrixBasedSolver,
	    AsymmetricMatrixBasedSolver,
	    SymmetricMatrixBasedSolver,
	    SolutionBasedSolver,
	)

	# Public classes and functions. Order is reflected in documentation.
	__all__ = [
	    "problinsolve",
	    "bayescg",
	    "ProbabilisticLinearSolver",
	    "MatrixBasedSolver",
	    "AsymmetricMatrixBasedSolver",
	    "SymmetricMatrixBasedSolver",
	    "SolutionBasedSolver",
	]

	# Set correct module paths. Corrects links and module paths in documentation.
	ProbabilisticLinearSolver.__module__ = "probnum.linalg"
	MatrixBasedSolver.__module__ = "probnum.linalg"


If you are documenting a subclass, which has a different path in the file structure than the import path due to
:code:`__all__` statements, you can correct the links to superclasses in the documentation via the :code:`.__module__` attribute.