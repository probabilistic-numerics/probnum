# Contributing via Pull Request

Further pointers:

- 
- Check the code style guide (See here, how to auto-format your code with tox). 
- where applicalble, please add test




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


