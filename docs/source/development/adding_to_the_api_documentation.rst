Adding to the API Documentation
===============================

Documentation is an integral part of every collaborative software
project. Good documentation not only encourages users of the package to
try out different functionalities, but it also makes maintaining and
expanding code significantly easier. Every code contribution to the
package must come with appropriate documentation of the API. This guide
details how to do this.

Docstrings
----------

The main form of documentation are docstrings, multi-line comments
beneath a class or function definition with a specific syntax, which
detail its functionality. This package uses the `NumPy docstring
format <https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide%3E>`__.
As a rule, all functions which are exposed to the user *must* have
appropriate docstrings. Below is an example of a docstring for a
probabilistic numerical method.

.. code:: ipython3

    # %load -r 1-163 ../../../src/probnum/linalg/_problinsolve.py
    """Probabilistic numerical methods for solving linear systems.
    
    This module provides routines to solve linear systems of equations in a
    Bayesian framework. This means that a prior distribution over elements
    of the linear system can be provided and is updated with information
    collected by the solvers to return a posterior distribution.
    """
    
    import warnings
    from typing import Callable, Dict, Optional, Tuple, Union
    
    import numpy as np
    import scipy.sparse
    
    import probnum  # pylint: disable=unused-import
    from probnum import linops, randvars, utils
    from probnum.linalg.solvers.matrixbased import SymmetricMatrixBasedSolver
    from probnum.typing import LinearOperatorLike
    
    # pylint: disable=too-many-branches
    
    
    def problinsolve(
        A: Union[
            LinearOperatorLike,
            "randvars.RandomVariable[LinearOperatorLike]",
        ],
        b: Union[np.ndarray, "randvars.RandomVariable[np.ndarray]"],
        A0: Optional[
            Union[
                LinearOperatorLike,
                "randvars.RandomVariable[LinearOperatorLike]",
            ]
        ] = None,
        Ainv0: Optional[
            Union[
                LinearOperatorLike,
                "randvars.RandomVariable[LinearOperatorLike]",
            ]
        ] = None,
        x0: Optional[Union[np.ndarray, "randvars.RandomVariable[np.ndarray]"]] = None,
        assume_A: str = "sympos",
        maxiter: Optional[int] = None,
        atol: float = 10 ** -6,
        rtol: float = 10 ** -6,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[
        "randvars.RandomVariable[np.ndarray]",
        "randvars.RandomVariable[linops.LinearOperator]",
        "randvars.RandomVariable[linops.LinearOperator]",
        Dict,
    ]:
        r"""Solve the linear system :math:`A x = b` in a Bayesian framework.
    
        Probabilistic linear solvers infer solutions to problems of the form
    
        .. math:: Ax=b,
    
        where :math:`A \in \mathbb{R}^{n \times n}` and :math:`b \in \mathbb{R}^{n}`.
        They return a probability measure which quantifies uncertainty in the output arising
        from finite computational resources or stochastic input. This solver can take prior
        information either on the linear operator :math:`A` or its inverse :math:`H=A^{
        -1}` in the form of a random variable ``A0`` or ``Ainv0`` and outputs a posterior
        belief about :math:`A` or :math:`H`. This code implements the method described in
        Wenger et al. [1]_ based on the work in Hennig et al. [2]_.
    
        Parameters
        ----------
        A
            *shape=(n, n)* -- A square linear operator (or matrix). Only matrix-vector
            products :math:`v \mapsto Av` are used internally.
        b
            *shape=(n, ) or (n, nrhs)* -- Right-hand side vector, matrix or random
            variable in :math:`A x = b`.
        A0
            *shape=(n, n)* -- A square matrix, linear operator or random variable
            representing the prior belief about the linear operator :math:`A`.
        Ainv0
            *shape=(n, n)* -- A square matrix, linear operator or random variable
            representing the prior belief about the inverse :math:`H=A^{-1}`. This can be
            viewed as a preconditioner.
        x0
            *shape=(n, ) or (n, nrhs)* -- Prior belief for the solution of the linear
            system. Will be ignored if ``Ainv0`` is given.
        assume_A
            Assumptions on the linear operator which can influence solver choice and
            behavior. The available options are (combinations of)
    
            ====================  =========
             generic matrix       ``gen``
             symmetric            ``sym``
             positive definite    ``pos``
             (additive) noise     ``noise``
            ====================  =========
    
        maxiter
            Maximum number of iterations. Defaults to :math:`10n`, where :math:`n` is the
            dimension of :math:`A`.
        atol
            Absolute convergence tolerance.
        rtol
            Relative convergence tolerance.
        callback
            User-supplied function called after each iteration of the linear solver. It is
            called as ``callback(xk, Ak, Ainvk, sk, yk, alphak, resid, **kwargs)`` and can
            be used to return quantities from the iteration. Note that depending on the
            function supplied, this can slow down the solver considerably.
        kwargs
            Optional keyword arguments passed onto the solver iteration.
    
        Returns
        -------
        x :
            Approximate solution :math:`x` to the linear system. Shape of the return matches
            the shape of ``b``.
        A :
            Posterior belief over the linear operator.
        Ainv :
            Posterior belief over the linear operator inverse :math:`H=A^{-1}`.
        info :
            Information on convergence of the solver.
    
        Raises
        ------
        ValueError
            If size mismatches detected or input matrices are not square.
        LinAlgError
            If the matrix ``A`` is singular.
        LinAlgWarning
            If an ill-conditioned input ``A`` is detected.
    
        Notes
        -----
        For a specific class of priors the posterior mean of :math:`x_k=Hb` coincides with
        the iterates of the conjugate gradient method. The matrix-based view taken here
        recovers the solution-based inference of :func:`bayescg` [3]_.
    
        References
        ----------
        .. [1] Wenger, J. and Hennig, P., Probabilistic Linear Solvers for Machine Learning,
           *Advances in Neural Information Processing Systems (NeurIPS)*, 2020
        .. [2] Hennig, P., Probabilistic Interpretation of Linear Solvers, *SIAM Journal on
           Optimization*, 2015, 25, 234-260
        .. [3] Bartels, S. et al., Probabilistic Linear Solvers: A Unifying View,
           *Statistics and Computing*, 2019
    
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
        9
        """

**General Rules**

-  Cover ``Parameters``, ``Returns``, ``Raises`` and ``Examples``, if
   applicable, in every publicly visible docstring—in that order.
-  Examples are tested via doctest. Ensure ``doctest`` does not fail by
   running the test suite.
-  Include appropriate ``References``, in particular for probabilistic
   numerical methods.
-  Do not use docstrings as a clutch for spaghetti code!

**Parameters**

-  Parameter types are automatically documented via type hints in the
   function signature.
-  Always provide shape hints for objects with a ``.shape`` attribute in
   the following form:

.. code:: python

   """
   Parameters
   ----------
   arr :
       *(shape=(m, ) or (m, n))* -- Parameter array of an example function.
   """

-  Hyperparameters should have default values and explanations on how to
   choose them.
-  For callables provide the expected signature as part of the
   docstring: ``foobar(x, y, z, \*\*kwargs)``. Backslashes remove
   semantic meaning from special characters.

**Style**

-  Stick to the imperative style of writing in the docstring header
   (i.e.: first line).

   -  Yes: “Compute the value”.
   -  No: “This function computes the value / Let’s compute the value”.

   The rest of the explanation talks about the function, e. g. “This
   function computes the value by computing another value”.
-  Use full sentences inside docstrings when describing something.

   -  Yes: “This value is irrelevant, because it is not being passed on”
   -  No: “Value irrelevant, not passed on”.

-  When in doubt, more explanation rather than less. A little text
   inside an example can be helpful, too.
-  A little maths can go a long way, but too much usually adds
   confusion.

Interface Documentation
-----------------------

Which functions and classes actually show up in the documentation is
determined by an ``__all__`` statement in the corresponding
``__init__.py`` file inside a module. The order of this list is also
reflected in the documentation. For example, ``linalg`` has the
following ``__init__.py``:

.. code:: ipython3

    # %load ../../../src/probnum/linalg/__init__.py
    """Linear Algebra.
    
    This package implements probabilistic numerical methods for the solution of problems
    arising in linear algebra, such as the solution of linear systems :math:`Ax=b`.
    """
    from probnum.linalg._bayescg import bayescg
    from probnum.linalg._problinsolve import problinsolve
    
    # Public classes and functions. Order is reflected in documentation.
    __all__ = [
        "problinsolve",
        "bayescg",
    ]
    
    # Set correct module paths. Corrects links and module paths in documentation.
    problinsolve.__module__ = "probnum.linalg"
    bayescg.__module__ = "probnum.linalg"
    

If you are documenting a subclass, which has a different path in the
file structure than the import path due to ``__all__`` statements, you
can correct the links to superclasses in the documentation via the
``.__module__`` attribute.

Sphinx
------

ProbNum uses `Sphinx <https://www.sphinx-doc.org/en/master/>`__ to parse
docstrings in the codebase automatically and to create its API
documentation. You can configure Sphinx itself or its extensions in the
``./docs/conf.py`` file.

.. code:: ipython3

    from IPython.display import Image
    
    display(Image(filename="../assets/img/developer_guides/sphinx_logo.png", embed=True))



.. image:: ../assets/img/developer_guides/sphinx_logo.png


ProbNum makes use of a number of Sphinx plugins to improve the API
documentation, for example to parse this Jupyter notebook. The full list
of used packages can be found in ``./docs/sphinx-requirements.txt`` and
``./docs/notebook-requirements.txt``.

Building and Viewing the Documentation
--------------------------------------

In order to build the documentation locally and view the HTML version of
the API documentation, simply run:

.. code:: bash

   tox -e docs

This creates a static web page under ``./docs/_build/html/`` which you
can view in your browser by opening ``./docs/_build/html/intro.html``.

Alternatively, if you want to build the docs in your current environment
you can manually execute

.. code:: bash

   cd docs
   make clean
   make html

For more information on ``tox``, check out the `general development
instructions <../development/pull_request.md>`__.
