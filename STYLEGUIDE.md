# Style Guide for Code Contributions

This style guide summarizes code conventions used in `probnum`. This is intended as a reference for developers. The primary standard that should be applied is [PEP 8](https://www.python.org/dev/peps/pep-0008/). Use a linter (pylint, flake8) if necessary.
As of now the documentation is written in American English (although this is not 100% consistent through the code base).

## Code

### Imports

Use absolute imports over relative imports.

- `import x` for importing packages and modules.
- `from x import y` where `x` is the package prefix and `y` is the module name with no prefix.
- `from x import y as z` if two modules named `y` are to be imported or if `y` is an inconveniently long name.
- `import y as z` only when `z` is a standard abbreviation (e.g., `np` for `numpy`).

Use `__all__ = [...]` in `__init__.py` files to imply the order in which the methods are visible in the documentation.
Almost all methods are "pulled up" to a higher-level namespace. Import from there wherever there is no chance for confusion and/or circular imports. This makes imports more readable.


### Naming

High level convenience functions attempt to replicate scipy naming conventions.
For example
- `probsolve_ivp(...)` (scipy: `solve_ivp(...)`)
- `probsolve(...)` (scipy: `solve(...)`)

#### Modules and Folders
- `low` (shortened lower caps) for modules/folders in the namespace, e.g. `probnum.linalg.linops`
- `lower` for modules/folders not in the namespace, e.g. `probnum/prob/distributions`.
*Rule of thumb:* the more low-level the module is, the longer
(more descriptive) the file name can be, because the chances
that access is provided through higher-level namespaces are rather high.

PN methods should be in a file with the same name as the containing folder (e.g. `probnum/linalg/linearsolvers/linearsolvers.py`), while their implementation (in classes) is in the same folder in other files.

#### Classes, Functions, Methods and Variables
- `joined_lower` for functions, methods, attributes, variables
- `joined_lower` or `ALL_CAPS` for constants
- `StudlyCaps` for classes
- `camelCase` only to conform to pre-existing conventions, e.g. in `unittest`


### Printable Representations
The way an object is printed defined by `def __repr__(self)` is structured as:
```

```

### Other Notational Conventions
Generally, we tend to stick to the first few letters for abbreviations
if they are sufficiently desciptive. E.g.:
- `cov`: covariance
- `fun`: function
- `mat`: matrix
- `vec`: vector
- `arr`: array; wherever applicable, specify `vec` or `mat`

Further conventions are
- `unit2unit`: convert between types or units; e.g. 
`mat2arr`: convert matrix to array or `s2ms`: convert seconds to miliseconds.
Can also be used for simple adapter methods, along the lines of `filt2odefilt`.
- `proj`: projection (if required: `projmat`, `projvec`, `projlinop`, ...)
- `precond`: preconditioner
- `inv*`: for inverse of matrix; e.g. `invprecond`, `invcovmat`, ...
- optional arguments via `**kwargs`, e.g.: `fun(t, x, **kwargs)`
- `msg`: message, e.g. for issuing raising and warnings (`errmsg`, `warnmsg`)
- `randvar`: random variable; if concatenated with e.g. `init`, abbreviate to `initrv` (initial random variable)
- `dist`: distribution. The only exception is in instantiation of a
  `RandomVariable` object, where specifies a `distribution` key.
- `data`: data (don't abbreviate that one)
- functions/methods that do something *from* time `t0` *to* time `t1`
  with step size `h` use the signature `(start, stop, step, **kwargs)`
  or any corresponding subset of that. This is in line with `np.arange`
  for instance. Use it like `(start=t0, stop=t1, step=h, **kwargs)`.
- methods with "Bayesian" in the name come with the prefix `bayes`,
  e.g. `bayesquad`: Bayesian quadrature, `BayesFilter`: Bayesian filter,
  `BayesSmoother`: Bayesian smoother
- `jacob`: Jacobian, if necessary use `jacobfun`.

## Errors and Warnings
- Stick to the built-in python exceptions (`TypeError`, `NotImplementedError`, ...)
- If dunder method is not implemented for a type, return `NotImplemented`
- Warnings via `warnings.warn()`. See https://docs.python.org/2/library/warnings.html
or https://docs.python.org/2/library/exceptions.html#exceptions.Warning.
- recall the difference between TypeError and ValueError


    TypeError is thrown when an operation or function is applied to an object of an inappropriate type.
    ValueError is thrown when a function's argument is of an inappropriate type.


## Documentation

### Docstrings

This package uses the [`numpy` docstring format](https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide). For probabilistic numerical methods make sure to include the appropriate citations and include examples which can be used as tests via `doctest`. Here is a detailed example of a docstring for a PN method.

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

**Some objectives for writing docstrings**

* Do your best to cover "Parameters", "Returns", and "Examples" at every publicly visible docstring---in that order.
* Examples are tested via doctest. Bear that in mind when writing examples.
* When in doubt, more explanation rather than less.
* A little maths goes a long way.
* References make everything easier for someone who doesn't know the details of an algorithm.
* Parameters which are to be chosen benefit from a rule of thumb of how to choose it, perhaps even why.
* `np.ndarray`'s are documented as `array_like` (this is whay `numpy` does).
  Mention its shape in the docstring via `bla : array_like, shape=(a, b)` wherever possible (see example above)
* Callables are use as parameters if the expected signature is part of the docstring:
  `bla2 : callable, signature=``(t, **kwargs)`` `. The main effect of the double apostrophes is
  that the `**kwargs` does not raise a warning during building the documentation.

### Example Notebooks

Functionality of `probnum` is explained in detail in the form of `jupyter` notebooks under `/docs/source/notebooks`. 
These can be added to the documentation by 
1. Creating a `.ipynb` notebook in `/docs/source/notebooks/`.
2. Adding the notebook in the appropriate section in `/docs/source/notebooks/examples.rst`.

Note that notebooks must start with a upper level markdown header (`# Header`) to be converted correctly.