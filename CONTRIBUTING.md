# Developer Introduction

Contributions to ProbNum are very welcome. Before getting started make sure to read the following guidelines.

All contributions to ProbNum should be made via pull requests (PR) to the
[master branch](https://github.com/probabilistic-numerics/probnum/tree/master) on GitHub. Some suggestions for
a good PR are:

- implements or fixes one functionality;
- includes tests and appropriate documentation; and
- makes minimal changes to the interface and core codebase.

If you would like to contribute but are unsure how, then writing examples, documentation or working on
[open issues](https://github.com/probabilistic-numerics/probnum/issues) are a good way to start. See the
[developer guides](https://probnum.readthedocs.io/en/latest/development/developer_guides.html)
for detailed instructions.

## Getting Started

Begin by forking the repository on GitHub and cloning your fork to a local machine.

```shell
git clone git@github.com:MyGithubAccount/probnum.git
```

Next, create a new branch in your
forked repository describing the feature you would like to implement.
```shell
git checkout -b my-new-feature
```
You can now get started writing code in your new
branch. Make sure to keep the following best practices regarding code quality in mind.

### Code Quality

Code quality is an essential component in a collaborative open-source project.

- Make sure to observe [good coding practice](https://www.python.org/dev/peps/pep-0020/).
- Keep dependencies to a minimum.
- All code should be covered by tests within the [pytest](https://docs.pytest.org/) framework.
- Documentation of code is essential. ProbNum uses the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Code should be formatted with [*Black*](https://github.com/psf/black) and follow the internal [style guide](https://github.com/probabilistic-numerics/probnum/blob/master/STYLEGUIDE.md).
  For more thorough Python code style guides we refer to [PEP 8](https://www.python.org/dev/peps/pep-0008/) and to [the *Black* code style](https://github.com/psf/black/blob/master/docs/the_black_code_style.md).

For all of the above the existing ProbNum code is a good initial reference point.

### Using Tox

Probnum uses [tox](https://tox.readthedocs.io/en/latest/) in its [continuous integration (CI)](#continuous-integration)
pipeline to run tests, build documentation, check code formatting and code quality. Under the hood, tox builds virtual
environments following the specifications in `./tox.ini` in order to run tests across multiple python versions, while
making sure that all the necessary dependencies are installed. Using tox unifies the
*local* development process with continuous integration builds (via Github Actions),
such that local test results should match the outcomes of the CI builds more closely.
This ensures that your pull request can be merged seamlessly into ProbNum's codebase.

Install tox from the Python Package Index (PyPI) via
```shell
pip install -U tox
```
Once tox and the required [external tools](#external-tools) below are installed, you can run tests and build the
documentation locally by simply calling
```shell
tox
```
Note, to reduce runtime tox caches and reuses the virtual environment it creates the first time you run the command. If
you are frequently switching between branches or adjusting the build configuration make sure to force recreation of the
virtual environment via ``tox -r``, if you experience unexpected tox failures.

**Word of caution:**
Running `tox` runs all environments as specified in `tox.ini`, thus potentially running tests across many different
Python versions. To run the full test suite make sure that you have all specified Python versions installed.
Alternatively, you can run a single specific environment through `tox -e <env>`, e.g. `tox -e py36` to run tests with
Python 3.6 or `tox -e docs` to just build the documentation.

### External Tools

Building the documentation locally requires additional packages (e.g. for inheritance diagrams), which can be found in
`.github/workflows/CI-build.yml`. These packages are currently:
- [pandoc](https://pandoc.org/): In Ubuntu, install via `sudo apt install pandoc`
- [graphviz](https://graphviz.org/): In Ubuntu, install via `sudo apt install graphviz`

## Testing

[![test coverage: latest](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?label=Coverage%3A%20latest&logo=codecov)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/master)

We use [pytest](https://docs.pytest.org/) for testing and aim to cover as much code with tests
as possible. Make sure to always add tests for newly implemented code. To run the test suite on your machine you have
multiple options:

- **Full test suite with tox:** Run the full suite across different Python versions with

  ```shell
  tox
  ```

- **Single environment with tox:** Run tests for a single Python environment, e.g. for Python 3.6

  ```shell
  tox -e py36
  ```

- **pytest:** Run tests directly in your local environment by calling

  ```shell
  pytest
  ```

Code coverage of the tests is reported via [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master).

## Documentation

[![docs: stable](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Docs:%20stable)](https://probnum.readthedocs.io/en/stable/)
[![docs: latest](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Docs:%20latest)](https://probnum.readthedocs.io/en/latest/)

ProbNum's documentation is created with [Sphinx](https://www.sphinx-doc.org/en/master/) and automatically built and
hosted by [ReadTheDocs](https://readthedocs.org/projects/probnum/) for stable releases and the latest (`master` branch)
version.

You can build the documentation locally via
```shell
tox -e docs
```
This creates a static web page under `./docs/_build/html/` which you can view in your browser by opening
`./docs/_build/html/intro.html`.

Alternatively, if you want to build the docs in your current environment you can manually execute
```shell
cd docs
make clean
make html
```

## Continuous Integration

[![CI build](https://img.shields.io/github/workflow/status/probabilistic-numerics/probnum/Linting?logo=github&logoColor=white&label=CI-build)](https://github.com/probabilistic-numerics/probnum/actions?query=workflow%3ACI-build)

ProbNum uses [Github Actions](https://github.com/probabilistic-numerics/probnum/actions) for continuous integration.
For every pull request and every commit the project is built, the test suite is run,
the documentation is built, the benchmarks are dry-run, the code is linted and
checked for consistency with the [*Black*](https://github.com/psf/black) code style.
This ensures that no breaking changes are introduced by mistake. Changes to
Github Actions can be made in the  `.github/workflows/` folder, as well as in
`tox.ini` since Github Actions rely on `tox` for all the above checks.


## Advanced Developer Setup

If you regularly write code for ProbNum, the following additional development setup
recommendations might be useful.

### Multiple Remotes
In order to keep your branch with a new feature up-to-date with the main repository, one convenient
way to do so is to set up [multiple remotes](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)
for `git`.
For example, this way you can always keep the master branch of your forked repository
up-to-date with the main repository and you can easily `git merge` recent changes into
your feature branch. To set up the main ProbNum repository as a secondary remote run

```shell
git remote add probabilistic-numerics https://github.com/probabilistic-numerics/probnum
```

Now in addition to `origin`, which is your fork of ProbNum, you have access to the main
repository.

```shell
git remote -v
```

Now to get the latest changes from the `master` branch of the main repository simply run:
```shell
git checkout master
git pull probabilistic-numerics master
```

### Pre-commit Hooks

Pre-commit hooks allow the automatic identification of simple issues in a commit, e.g.
inconsistent code formatting. They are
executed automatically whenever `git commit` is executed. This way one can avoid common
problems in a pull request which prevent an automatic merge into the `master` branch on
GitHub. To set up ProbNum's pre-commit hooks simply install [pre-commit](https://pre-commit.com/) by executing
```shell
pip install pre-commit
```
and install the provided configuration file `.pre-commit-config.yaml ` with a recommended
set of pre-commit hooks via
```shell
pre-commit install
```
in the `probnum` folder.
