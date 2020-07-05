# Adding to the Documentation

Documentation is an integral part of every collaborative software project. Good documentation not only encourages users of the package to try out different functionalities, but it also makes maintaining and expanding code significantly easier. If you want to improve ProbNum's documentation or learn how to write documentation for your newly implemented functionality, keep reading.

## Docstrings

The main form of documentation are `docstrings`, multi-line comments beneath a class or function definition with a specific syntax, which detail its functionality. This package uses the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide). As a rule, all functions which are exposed to the user _must_ have appropriate docstrings. Below is an example of a docstring for a probabilistic numerical method.

```python
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

```

For probabilistic numerical methods make sure to include the appropriate citations and include examples which are checked for correctness via `doctest`. 


**Some objectives for writing docstrings**

* Use full sentences inside docstrings when describing something.
  Yes: "This value is irrelevant, because it is not being passed on"
  No: "Value irrelevant, not passed on". 
* Stick to the imperative style of writing in the "main sentence"
  of the docstring (i.e.: first line).
  Yes: "Compute the value". No:"This function computes the value / Let's compute the value".
  The rest of the explanation talks "about" the function.
  Yes: "This function computes the value by computing another value".
* Do your best to cover "Parameters", "Returns", and "Examples" at every publicly visible docstring---in that order.
* Examples are tested via doctest. Bear that in mind when writing examples.
* When in doubt, more explanation rather than less. A little text inside an example can be helpful, too.
* A little maths goes a long way.
* References make everything easier for someone who doesn't know the details of an algorithm.
* Parameters which are to be chosen benefit from a rule of thumb of how to choose it, perhaps even why.
* `np.ndarray`'s are documented as `array_like` (this is whay `numpy` does).
  Mention its shape in the docstring via `bla : array_like, shape=(a, b)` wherever possible (see example above)
* Callables are use as parameters if the expected signature is part of the docstring:
  `bla2 : callable, signature=``(t, **kwargs)`` `. The main effect of the double apostrophes is
  that the `**kwargs` does not raise a warning during building the documentation.

