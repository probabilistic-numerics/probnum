# Package Development

Contributions to ProbNum are very welcome. Before getting started make sure to read the following guidelines.

## In a Nutshell

All contributions to ProbNum should be made via pull requests (PR) to the
[development branch](https://github.com/probabilistic-numerics/probnum/tree/development) on GitHub. Some suggestions for
a good PR are:

- implements or fixes one functionality
- includes tests and appropriate documentation
- makes minimal changes to the interface and core codebase

If you would like to contribute but are unsure how, then writing examples, documentation or working on
[open issues](https://github.com/probabilistic-numerics/probnum/issues) are a good way to start. See the [detailed contribution guide](https://probabilistic-numerics.github.io/probnum/development/contributing.html#detailed-contribution-guide) for more instructions.

### Code Quality

Code quality is an essential component in a collaborative open-source project.

- All code should be covered by tests within the [unittest](https://docs.python.org/3/library/unittest.html) framework. Every time a commit is
made [Travis](https://travis-ci.org/probabilistic-numerics/probnum) builds the project and runs the test suite.
- Documentation of code is essential for any collaborative project. ProbNum uses the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Code should follow the [PEP8 style](https://www.python.org/dev/peps/pep-0008/) and the internal [style guide](https://github.com/probabilistic-numerics/probnum/blob/master/STYLEGUIDE.md).
- Keep dependencies to a minimum.
- Make sure to observe good coding practice.

For all of the above the existing ProbNum code is a good reference.

### Tests and CI

We aim to cover as much code with tests as possible. Make sure to add tests for newly implemented code. Tests are run by
the continuous integration tool [Travis](https://travis-ci.org/probabilistic-numerics/probnum) and coverage is reported
by [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master). If you cloned this repository, you
can run the test suite on your machine via:
```bash
cd probnum
pytest
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


## Detailed Contribution Guide

### Setting up the Development Environment


