# Contributing to Probnum

Contributions to Probnum are very welcome. Before getting started make sure to read the following notes.

* [Getting Started](#getting-started)
* [Code Quality](#code-quality)
* [Tests and CI](#tests-ci)
* [Documentation](#documentation)

## <a name="getting-started">Getting Started</a>

All contributions to Probnum should be made via pull requests (PR) to the development branch. Some suggestions for a 
good PR are:

- implements or fixes one functionality
- includes tests and appropriate documentation
- makes minimal changes to the interface and core codebase

If you would like to contribute but are unsure how, then writing examples, documentation or `jupyter` notebooks are a 
good way to start.

## <a name="code-quality">Code Quality</a>

Code quality is an essential component in a collaborative open-source project.

- All code should be covered by tests. We use the [pytest](https://docs.pytest.org/) framework. Every time a commit is 
made `travis` builds the project and runs the test suite.
- Documentation of code is essential for any collaborative project. Probnum uses the 
[`numpy` docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Python code should follow the [*PEP8* style](https://www.python.org/dev/peps/pep-0008/). 
- Make sure to observe good coding practice. The existing ProbNum code is a good starting point for coding style.

Documentation example:

```python
def some_function(param1, param2):
    """Short function description.

    Extensive function description which explains its purpose in more detail and may reference parameters or output. 
    References [1]_ can also be included, e.g. if the method implementation is based on a paper.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : float
        The second parameter.

    Returns
    -------
    float
        Sum of the input parameters.

    References
    ----------
    .. [1] Some paper title, A. Author, EXAMPLE conference 2019
    """
    return param1 + param2
```

## <a name="tests-ci">Tests and CI</a>

We aim to cover as much code with tests as possible. Make sure to add tests for newly implemented code. Tests are run by 
the continuous integration (CI) tool `travis` and coverage is reported by `codecov`.

## <a name="documentation">Documentation</a>

[Documentation](https://readthedocs.org/probabilistic-numerics/probnum) is automatically built using `sphinx` and 
`travis`. When implementing published probabilistic numeric methods please give credit and include citations. 
