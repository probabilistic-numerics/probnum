# Package Development

Contributions to ProbNum are very welcome. Before getting started make sure to read the following guidelines.

All contributions to ProbNum should be made via pull requests (PR) to the
[development branch](https://github.com/probabilistic-numerics/probnum/tree/development) on GitHub. Some suggestions for
a good PR are:

- implements or fixes one functionality;
- includes tests and appropriate documentation; and
- makes minimal changes to the interface and core codebase.

If you would like to contribute but are unsure how, then writing examples, documentation or working on
[open issues](https://github.com/probabilistic-numerics/probnum/issues) are a good way to start. See the
[developer guides](https://probnum.readthedocs.io/en/latest/development/developer_guides.html)
for detailed instructions.

## Getting Started

Begin by forking the repository on GitHub and cloning your fork to a local machine. Next, create a new branch in your forked repository describing the feature you would like to implement. You can now get started writing code in your new branch. Make sure to keep the following best practices regarding code quality in mind.

### Code Quality

Code quality is an essential component in a collaborative open-source project.

- Make sure to observe [good coding practice](https://www.python.org/dev/peps/pep-0020/).
- Keep dependencies to a minimum.
- All code should be covered by tests within the [unittest](https://docs.python.org/3/library/unittest.html) framework.
- Documentation of code is essential. ProbNum uses the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Code should be formatted with [*Black*](https://github.com/psf/black) and follow the internal [style guide](https://github.com/probabilistic-numerics/probnum/blob/master/STYLEGUIDE.md).
  For more thorough Python code style guides we refer to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and to [the *Black* code style](https://github.com/psf/black/blob/master/docs/the_black_code_style.md).

For all of the above the existing ProbNum code is a good initial reference point.

### Using Tox

Probnum uses [tox](https://tox.readthedocs.io/en/latest/) in its [continuous integration (CI)](#continuous-integration) pipeline to run tests and to build documentation. Under the hood, tox builds virtual environments following the specifications in `./tox.ini` in order to run tests across multiple python versions, while making sure that all the necessary dependencies are installed. Using tox unifies the *local* development process with CI, such that local test results should match the outcomes of Travis's builds more closely. This ensures that your pull request can be merged seemlessly into ProbNum's codebase.

Install tox from the Python Package Index (PyPI) via
```bash
pip install -U tox
```
Once tox and the required [external tools](#external-tools) below are installed, you can run tests and build the documentation locally by simply calling
```bash
tox
```

**Word of caution:**
Running `tox` runs all environments as specified in `tox.ini`, thus potentially running tests across many different python versions.
To run the full test suite make sure that you have all specified python versions installed.
Alternatively, you can run a single specific environment through `tox -e <env>`, e.g. `tox -e py36` to run tests with Python 3.6 or `tox -e docs` to just build the documentation.

### External Tools

Building the documentation locally requires additional packages (e.g. for inheritance diagrams), which can be found in `.travis.yml`. These packages are currently:
- [pandoc](https://pandoc.org/): In Ubuntu, install via `sudo apt install pandoc`
- [graphviz](https://graphviz.org/): In Ubuntu, install via `sudo apt install graphviz`

## Testing

[![test coverage: latest](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?label=Coverage%3A%20latest&logo=codecov)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/master)
[![test coverage: development](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/development?label=Coverage%3A%20dev&logo=codecov)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/development)

We use [unittest](https://docs.python.org/3/library/unittest.html) for testing and aim to cover as much code with tests as possible. Make sure to always add tests for newly implemented code. To run the test suite on your machine you have multiple options:

- **Full test suite with tox:** Run the full suite across different Python versions with
  
  ```bash
  tox
  ```
  
- **Single environment with tox:** Run tests for a single Python environment, e.g. for Python 3.6
  
  ```bash
  tox -e py36
  ```
  
- **pytest:** Run tests directly in your local environment by calling
  
  ```bash
  pytest
  ```

## Documentation

[![docs: stable](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Docs:%20stable)](https://probnum.readthedocs.io/en/stable/)
[![docs: latest](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Docs:%20latest)](https://probnum.readthedocs.io/en/latest/)
[![docs: development](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Docs:%20dev&version=development)](https://probnum.readthedocs.io/en/development/)

ProbNum's documentation is created with [Sphinx](https://www.sphinx-doc.org/en/master/) and automatically built and hosted by [ReadTheDocs](https://readthedocs.org/projects/probnum/) for stable (release), latest (`master` branch) and dev (`development` branch) versions.

You can build the documentation locally via
```bash
tox -e docs
```
This creates a static web page under `./docs/_build/html/` which you can view in your browser by opening `./docs/_build/html/intro.html`.

Alternatively, if you want to build the docs in your current environment you can manually execute
```bash
cd docs
make clean
make html
```

## Continuous Integration

[![build status: latest](https://img.shields.io/travis/probabilistic-numerics/probnum/master.svg?logo=travis%20ci&logoColor=white&label=Travis%20CI:%20latest)](https://travis-ci.org/probabilistic-numerics/probnum)
[![build status: development](https://img.shields.io/travis/probabilistic-numerics/probnum/development.svg?logo=travis%20ci&logoColor=white&label=Travis%20CI:%20dev)](https://travis-ci.org/probabilistic-numerics/probnum)

ProbNum uses [Travis CI](https://travis-ci.org/probabilistic-numerics/probnum) for continuous integration.
For every pull request and every commit Travis builds the project and runs the test suite (through `tox`), to make sure that no breaking changes are introduced by mistake.
Code coverage of the tests is reported through [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master). Travis also automatically triggers a build of ProbNum's documentation. Changes to Travis can be made through the `.travis.yml` file, as well as through `tox.ini` since Travis relies on `tox` for both testing and building the documentation.
ProbNum also uses [GitHub Actions](https://docs.github.com/en/actions) to verify that all pushes and pull requests are compliant with the [*Black*](https://github.com/psf/black) code style.
