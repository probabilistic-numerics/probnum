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

Use `__all__ = [...]` in `__init__.py` files to imply the order in
which the methods are visible in the documentation.
It also helps with imports such as ``from ... import *`` by avoiding
importing private variables and constants.
Almost all methods are "pulled up" to a higher-level namespace.
Import from there wherever there is no chance for confusion and/or circular imports. This makes imports more readable.


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
*Begin suggestion by N...*

The way an object is printed defined by `def __repr__(self)` is structured as:
```
<class 'RandomProcess(rvcoll=<class 'function'>, support=<class 'list'>, bounds=<class 'list'>)'>
<class 'RandomProcess(rvcoll=<class 'np.ndarray'>, support=<class 'list'>, bounds=<class 'list'>)'>
```
Essentially, this is similar to the output of `type(obj)`. 
The effect is that representation of an object provides another level of
documentation with keys and parameter types.

*... end suggestion*

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
- `jacob`: Jacobian, if necessary use `jacobfun`. Hessians are `hess`, respectively
  `hessfun`.
- `param(s)`: parameter(s). If abbreviations are necessary
  (e.g. in inline-function definition, use `par(s)`).
- Indices via `idx` (either `idx1`, `idx2`, ... or `idx`, `jdx`, `kdx`)
  and not via `i, j, k`. The former is more readable (and follows PEP8);
  the latter may collide with the built-in imaginary constant `j=sqrt(-1)`.
- A function maps from its ``domain`` to its ``range``.
  The ``range`` of a random variable
  is the ``domain`` of its distribution. 
  
  The ``range`` of a random process
  is the ``range`` of the associated random variables. The ``domain`` of a random
  process is either given by ``supportpts`` {t_1, t_2, ...} (discrete-time processes)
  or  ``bounds`` (a, b) (continuous-time processes).

## Errors and Warnings
- Stick to the built-in python exceptions (`TypeError`, `NotImplementedError`, ...)
- If dunder method is not implemented for a type, return `NotImplemented`
- Warnings via `warnings.warn()`. See https://docs.python.org/2/library/warnings.html
or https://docs.python.org/2/library/exceptions.html#exceptions.Warning.
- recall the difference between TypeError and ValueError


    TypeError is thrown when an operation or function is applied to an object of an inappropriate type.
    ValueError is thrown when a function's argument is of an inappropriate type.
