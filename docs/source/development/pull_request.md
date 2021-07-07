# Creating a Pull Request

All code contributions to ProbNum should be made via pull requests (PR) to the
[master branch](https://github.com/probabilistic-numerics/probnum/tree/master) on GitHub.

To ensure a smooth workflow, please keep PRs as compact as possible.
Each PR should only contain one enhancement at a time.
If you implemented multiple changes, split them into several PRs.


## Code Quality

Code quality is an essential component in a collaborative open-source project.

- Make sure to observe [good coding practice](https://www.python.org/dev/peps/pep-0020/).
- New dependencies should only be added with care.
- All code should be covered by tests within the [pytest](https://docs.pytest.org/) framework.
- Documentation of code is essential. ProbNum uses the
[NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html).
- Code follows the internal [style guide](https://github.com/probabilistic-numerics/probnum/blob/master/STYLEGUIDE.md).
- To make life more simple, code should be formatted with [*Black*](https://github.com/psf/black).


For more thorough Python code style guides please refer to [PEP 8](https://www.python.org/dev/peps/pep-0008/)
and to [the Black code style](https://github.com/psf/black/blob/master/docs/the_black_code_style.md).
For all the above, the existing ProbNum code is a good initial reference point.

## Black

[Black](https://github.com/psf/black) is a simple code formatter. Install Black with `pip`.
```shell
$ pip install black
```
Format a single Python file.

```shell
$ black my-file.py
```
Black takes care of most of the PEP8 formatting rules.
Formatting with Black can also be done via tox.

## Forking the Repo

In order to do pull requests, begin by forking the repository on GitHub.
Then, clone your fork to a local machine.

```shell
$ git clone git@github.com:MyGithubAccount/probnum.git
```
Any code changes should be done in your fork. Github as official guides on [how to fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo)
and [how to create a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Multiple Remotes
In order to keep your fork up-to-date with the main repository, one convenient
way to do so is to set up [multiple remotes](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes)
for `git`.
For example, this way you can always keep the master branch of your forked repository
up-to-date with the main repository and you can easily `git merge` recent changes into
your feature branch. To set up the main ProbNum repository as a secondary remote run

```shell
$ cd probnum
$ git remote add upstream https://github.com/probabilistic-numerics/probnum
```

Now in addition to `origin`, which is your fork of ProbNum, you have access to the main
repository named `upstream`. Check your remotes are set up properly by executing the following command.

```shell
$ git remote -v
origin	https://github.com/MyGithubAccount/probnum.git (fetch)
origin	https://github.com/MyGithubAccount/probnum.git (push)
upstream	https://github.com/probabilistic-numerics/probnum (fetch)
upstream	https://github.com/probabilistic-numerics/probnum (push)
```
Fetch and checkout the master branch of ProbNum.
Since there exists already a local `master` branch tracking the master branch of your fork, we will name this one
`probnum-master` which tracks the master branch of `upstream`.

```shell
$ git fetch upstream
$ git checkout -b probnum-master upstream/master
```

## Code Changes
Next, create a new branch describing the feature you would like to implement.
```shell
$ git checkout -b my-new-feature
```
You can now get started writing code in your new branch.

Commit your changes to this local branch. Make sure to keep the best practices regarding code quality in mind.
Next, set up a branch on your fork which is tracked by your local branch.
```shell
git push --set-upstream origin my-new-feature
```
Once in a while, merge changes of the upstream master branch into your local `my-new-feature` branch.
```shell
$ git checkout probnum-master
$ git pull
$ git checkout my-new-feature
$ git merge probnum-master
```
## Creating a Pull-Request
Once you are happy with your PR, make sure your branch is up-to-date with upstream `master`, and re-run all tests.
Then, from your local branch, push the changes to your remote branch
```shell
$ git push
```

and create the PR via the GitHub interface. There, briefly explain the changes. That's it!
If possible, please try running the test suite with tox or pytest before creating the PR.


## Virtual Environments

Virtual environments (venvs) help you to get a separate, clean installation of ProbNum.
Each venv uses a specific Python version.
Venvs are useful as you may need different Python versions and dependencies for different projects.
So, first make sure that you are using the correct Python version, and also that your pip
installation is up to date.

Install the [virtualenv](https://virtualenv.pypa.io/en/latest/) package.
```shell
$ pip install virtualenv
```

Go to the probnum root directory.
Then, create a virtual environment with the name `venv_probnum`.
This uses the Python version the `python` alias is linked to.
```shell
$ python -m venv venv_probnum
```


Activate the venv (the command below works for bash, other shells might require a different command).
```shell
$ source venv_probnum/bin/activate
```

Check if your venv uses the correct Python path and Python version.
(The Python path should show something like this `/home/MyUserName/probnum/venv_probnum/bin/python`.)
```shell
$ which python
$ python --version
```


You can deactivate the venv as well.
```shell
$ deactivate
```



Install ProbNum in your environment (activate it first).
```shell
$ pip install -e .
```

If needed, install additional requirements, e.g., for testing.
```shell
$ pip install -e .[test]
```

Check your installation with `pip freeze`.


It is possible to create virtual environments with most modern editors such as e.g., PyCharm. Alternatively, you can
add the venv you created above to your project in the editor.

## Testing
Tests are run with the [pytest package](https://docs.pytest.org/en/stable/).
Make sure ProbNum is installed including test requirements (`pip install -e .[test])`) in your virtual
environment; installing ProbNum that way will already have installed pytest.
(If you are unsure, use `pip freeze` to check the installation in your active venv).

Run tests with the Python version in the venv.
```shell
$ pytest
```
This should normally be enough to catch the biggest bugs. If you want to run the whole testsuite for several Python
versions, the preferred way is to use tox.


## tox

[![test coverage: latest](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?style=flat-square&label=Coverage%3A%20latest&logo=codecov)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/master)

Probnum uses [tox](https://tox.readthedocs.io/en/latest/) in its [continuous integration (CI)](#continuous-integration)
pipeline to run tests, build documentation, check code formatting and code quality. Under the hood, tox builds virtual
environments following the specifications in `./tox.ini` in order to run tests across multiple Python versions, while
making sure that all the necessary dependencies are installed. Using tox unifies the
local development process with continuous integration builds (via GitHub Actions),
such that local test results should match the outcomes of the CI builds more closely.
This ensures that your pull request can be merged seamlessly into ProbNum's codebase.

Install tox from the Python Package Index (PyPI) via
```shell
$ pip install -U tox
```
Some commands, such as building the documentation locally, requires additional packages
(e.g., for inheritance diagrams). These are
`.github/workflows/CI-build.yml`. These packages are currently:
- [pandoc](https://pandoc.org/): In Ubuntu, install via `sudo apt install pandoc`
- [graphviz](https://graphviz.org/): In Ubuntu, install via `sudo apt install graphviz`

In order to run tox smoothly, please install those as well.

Now, you can run all tests and build the documentation locally by simply calling
```shell
$ tox
```
To reduce runtime, tox caches and reuses the virtual environment it creates the first time you run the command. If
you are frequently switching between branches or adjusting the build configuration make sure to force recreation of the
virtual environment via ``tox -r``.


Additionally, the command `tox` runs *all environments* as specified in `tox.ini`, thus potentially running tests across many different
Python versions. To run the full test suite make sure that you have all specified Python versions installed.
Alternatively, you can run a single specific environment through `tox -e <env>`. Useful examples are:

- **Full test suite with tox:** Run the full suite across different Python versions with

  ```shell
  $ tox
  ```

- **Single environment with tox:** Run tests for a single Python environment, e.g. for Python 3.6

  ```shell
  $ tox -e py36
  ```

- **Single environment with tox:** only build the documentation.

  ```shell
  $ tox -e docs
  ```

Code coverage of the tests is reported via [codecov](https://codecov.io/github/probabilistic-numerics/probnum?branch=master).


## Documentation

[![docs: stable](https://img.shields.io/readthedocs/probnum.svg?style=flat-square&logo=read%20the%20docs&logoColor=white&label=Docs:%20stable)](https://probnum.readthedocs.io/en/stable/)
[![docs: latest](https://img.shields.io/readthedocs/probnum.svg?style=flat-square&logo=read%20the%20docs&logoColor=white&label=Docs:%20latest)](https://probnum.readthedocs.io/en/latest/)


ProbNum's documentation is created with [Sphinx](https://www.sphinx-doc.org/en/master/) and automatically built and
hosted by [ReadTheDocs](https://readthedocs.org/projects/probnum/) for stable releases and the latest (`master` branch)
version.



You can build the documentation locally via
```shell
$ tox -e docs
```
This creates a static web page under `./docs/_build/html/` which you can view in your browser by opening
`./docs/_build/html/intro.html`.

Alternatively, if you want to build the docs in your current environment you can manually execute
```shell
$ cd docs
$ make clean
$ make html
```

## Further Information

### Continuous Integration

[![CI build](https://img.shields.io/github/workflow/status/probabilistic-numerics/probnum/Linting?style=flat-square&logo=github&logoColor=white&label=CI-build)](https://github.com/probabilistic-numerics/probnum/actions?query=workflow%3ACI-build)

ProbNum uses [Github Actions](https://github.com/probabilistic-numerics/probnum/actions) for continuous integration.
For every pull request and every commit the project is built, the test suite is run,
the documentation is built, the benchmarks are dry-run, the code is linted and
checked for consistency with the [*Black*](https://github.com/psf/black) code style.
This ensures that no breaking changes are introduced by mistake. Changes to
Github Actions can be made in the  `.github/workflows/` folder, as well as in
`tox.ini` since Github Actions rely on `tox` for all the above checks.



### Pre-commit Hooks

If you regularly write code for ProbNum, pre-commit hoods might be useful.

Pre-commit hooks allow the automatic identification of simple issues in a commit, e.g.
inconsistent code formatting. They are
executed automatically whenever `git commit` is executed. This way one can avoid common
problems in a pull request which prevent an automatic merge into the `master` branch on
GitHub. To set up ProbNum's pre-commit hooks simply install [pre-commit](https://pre-commit.com/) by executing
```shell
$ pip install pre-commit
```
and install the provided configuration file `.pre-commit-config.yaml ` with a recommended
set of pre-commit hooks via
```shell
$ pre-commit install
```
in the `probnum` folder.
