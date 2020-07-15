# Package Development

Contributions to ProbNum are very welcome. Before getting started make sure to read the following guidelines.

All contributions to ProbNum should be made via pull requests (PR) to the
[development branch](https://github.com/probabilistic-numerics/probnum/tree/development) on GitHub. Some suggestions for
a good PR are:

- implements or fixes one functionality
- includes tests and appropriate documentation
- makes minimal changes to the interface and core codebase

If you would like to contribute but are unsure how, then writing examples, documentation or working on
[open issues](https://github.com/probabilistic-numerics/probnum/issues) are a good way to start. See the
[contribution tutorials](https://probabilistic-numerics.github.io/probnum/development/contributing.html#contribution-tutorials)
for detailed instructions.

## Getting Started

Begin by forking the repository on GitHub. You can then clone the fork to a local machine and get started.
ProbNum uses a set of packages (e.g. for documentation), which are not dependencies of the main package. In
order to be able to build the documentation make sure to have the packages listed in `.travis.yml` installed.

_Note_: Make sure your Sphinx version fulfills `sphinx >=1.7.5, <3`.

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

### Testing

We use [unittest](https://docs.python.org/3/library/unittest.html) for testing and aim to cover as much code with tests as possible.
Make sure to add tests for newly implemented code.
You can run the test suite on your machine via:
```bash
pytest
```

### Documentation

[Documentation](https://probabilistic-numerics.github.io/probnum/modules.html) is automatically built using [Sphinx](https://www.sphinx-doc.org/en/master/) and [Travis](https://travis-ci.org/probabilistic-numerics/probnum).
When implementing published methods give credit and include the appropriate citations.
You can build the documentation locally via:
```bash
cd docs
make clean
make html
```
This creates a static web page under `./docs/_build/html/` which you can view in your browser by opening `./docs/_build/html/intro.html`.


### Continuous Integration (CI)

ProbNum uses [Travis CI](https://travis-ci.org/probabilistic-numerics/probnum) for continuous integration.
For every pull request and every commit Travis builds the project and runs the test suite (through `tox`), to make sure that no breaking changes are introduced by mistake.
Code coverage of the tests is reported through [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master).
Travis also automatically builds and publishes the [ProbNum documentation](https://probabilistic-numerics.github.io/probnum/modules.html).

Changes to Travis can be made through the `.travis.yml` file, as well as through `tox.ini` since Travis relies on `tox` for both testing and building the documentation.
