# Package Development

Contributions to ProbNum are very welcome. Before getting started make sure to read the following guidelines.

## Getting Started

All contributions to ProbNum should be made via pull requests (PR) to the 
[development branch](https://github.com/probabilistic-numerics/probnum/tree/development) on GitHub. Some suggestions for 
a good PR are:

- implements or fixes one functionality
- includes tests and appropriate documentation
- makes minimal changes to the interface and core codebase

If you would like to contribute but are unsure how, then writing examples, documentation or working on
[open issues](https://github.com/probabilistic-numerics/probnum/issues) are a good way to start.

### Code Quality

Code quality is an essential component in a collaborative open-source project.

- All code should be covered by tests within the [unittest](https://docs.python.org/3/library/unittest.html) framework. Every time a commit is
made [Travis](https://travis-ci.org/probabilistic-numerics/probnum) builds the project and runs the test suite.
- Documentation of code is essential for any collaborative project. ProbNum uses the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Python code should follow the [PEP8 style](https://www.python.org/dev/peps/pep-0008/).
- Keep dependencies to a minimum.
- Make sure to observe good coding practice. The existing ProbNum code is a good starting point for coding style.

#### Docstring Example

```python
def some_function(param1, param2):
    """Short function description.

    Extensive function description which explains its purpose in more detail and may reference
    parameters or output. References [1]_ can also be included, e.g. if the method 
    implementation is based on a paper.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : float
        The second parameter.

    Returns
    -------
    sum : float
        Sum of the input parameters.

    References
    ----------
    .. [1] Author A. et al., Some paper title, *EXAMPLE conference* 2019
    """
    return param1 + param2
```

### Tests and CI

We aim to cover as much code with tests as possible. Make sure to add tests for newly implemented code. Tests are run by
the continuous integration tool [Travis](https://travis-ci.org/probabilistic-numerics/probnum) and coverage is reported 
by [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master). If you cloned this repository, you
can run the test suite on your machine via:
```bash
cd probnum
python -m unittest
```

### Documentation

[Documentation](https://probabilistic-numerics.github.io/probnum/modules.html) is automatically built using [Sphinx](https://www.sphinx-doc.org/en/master/) and
[Travis](https://travis-ci.org/probabilistic-numerics/probnum). When implementing published methods give credit and 
include the appropriate citations. You can build the documentation locally via:
```bash
cd probnum/docs
make clean
make html
```
For this to execute, make sure to have the appropriate documentation packages installed. A list can be found in `.travis.yml`. 

