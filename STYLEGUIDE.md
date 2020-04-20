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
- `lower` for modules/folders not in the namespace, e.g. `probnum/prob/distributions`

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
- `cov`: covariance
- `fun`: function
- `mtrx`: matrix
- `vec`: vector
- `arr`: array; wherever applicable, specify `vec` or `mtrx`
- `mtrx2arr` vs `mtrx_to_arr` ?
- optional arguments via `**kwargs`, e.g.: `fun(t, x, **kwargs)`


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
    A : array-like or LinearOperator, shape=(n,n)
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


### Example Notebooks

Functionality of `probnum` is explained in detail in the form of `jupyter` notebooks under `/docs/source/notebooks`. 
These can be added to the documentation by 
1. Creating a `.ipynb` notebook in `/docs/source/notebooks/`.
2. Adding the notebook in the appropriate section in `/docs/source/notebooks/examples.rst`.

Note that notebooks must start with a upper level markdown header (`# Header`) to be converted correctly.