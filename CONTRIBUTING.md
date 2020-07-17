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
Running tests on your machine and building the documentation locally requires additional setup.

### Required External Tools

Building the documentation locally requires additional packages, which can be found in `.travis.yml`.
At the time of writing, these packages are:
- [pandoc](https://pandoc.org/): In Ubuntu, install via `sudo apt install pandoc`
- [graphviz](https://graphviz.org/): In Ubuntu, install via `sudo apt install graphviz`

### tox

Probnum uses [tox](https://tox.readthedocs.io/en/latest/) through [Travis CI](#continuous-integration) to run tests and to build documentation.
Under the hood, tox builds virtual environments following the specifications in `./tox.ini` in order to run tests across multiple python versions, while making sure that all the necessary dependencies are installed.
Using tox unifies the *local* development process with CI, such that local test results should match the outcomes of Travis's builds more closely.

Tox can be installed directly from the Python Package Index (PyPI), e.g. through
```
pip install -U tox
```
Once tox and the required [external tools](#required-external-tools) are installed, you can run tests and build the documentation locally by simply calling
```
tox
```

**Word of caution:**
Running `tox` runs all environments as specified in `tox.ini`, thus potentially running tests across many different python versions.
To run the full test suite make sure that you have all specified python versions installed.
Alternatively, you can run a single specific environment through `tox -e <env>`, e.g. `tox -e py36` to run tests with Python 3.6 or `tox -e docs` to just build the documentation.

## Code Quality

Code quality is an essential component in a collaborative open-source project.

- All code should be covered by tests within the [unittest](https://docs.python.org/3/library/unittest.html) framework. Every time a commit is
made [Travis](https://travis-ci.org/probabilistic-numerics/probnum) builds the project and runs the test suite.
- Documentation of code is essential for any collaborative project. ProbNum uses the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Code should follow the [PEP8 style](https://www.python.org/dev/peps/pep-0008/) and the internal [style guide](https://github.com/probabilistic-numerics/probnum/blob/master/STYLEGUIDE.md).
- Keep dependencies to a minimum.
- Make sure to observe good coding practice.

For all of the above the existing ProbNum code is a good reference.

## Testing

We use [unittest](https://docs.python.org/3/library/unittest.html) for testing and aim to cover as much code with tests as possible.
Make sure to always add tests for newly implemented code.
To run the test suite on your machine you have multiple options:

- **Full test suite with tox:** Run the full suite across different Python versions with
  
  ```
  tox
  ```
- **Single environment with tox:** Run tests via [tox](https://probabilistic-numerics.github.io/probnum/development/contributing.html#tox) for a single Python environment. Example for Python 3.6:
  
  ```
  tox -e py36
  ```
- **pytest:** Run tests directly in your local environment by calling
  
  ```
  pytest
  ```

## Documentation

[Documentation](https://probabilistic-numerics.github.io/probnum/modules.html) is automatically built using [Sphinx](https://www.sphinx-doc.org/en/master/) and [Travis](https://travis-ci.org/probabilistic-numerics/probnum).
When implementing published methods give credit and include the appropriate citations.
You can build the documentation locally via:
```bash
tox -e docs
```
This creates a static web page under `./docs/_build/html/` which you can view in your browser by opening `./docs/_build/html/intro.html`.

Alternatively, you can manually call
```bash
cd docs
make clean
make html
```

## Continuous Integration (CI)

ProbNum uses [Travis CI](https://travis-ci.org/probabilistic-numerics/probnum) for continuous integration.
For every pull request and every commit Travis builds the project and runs the test suite (through `tox`), to make sure that no breaking changes are introduced by mistake.
Code coverage of the tests is reported through [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master).
Travis also automatically builds and publishes the [ProbNum documentation](https://probabilistic-numerics.github.io/probnum/modules.html).

Changes to Travis can be made through the `.travis.yml` file, as well as through `tox.ini` since Travis relies on `tox` for both testing and building the documentation.
