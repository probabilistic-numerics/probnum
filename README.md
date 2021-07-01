<div align="center">
    <a href="https://probnum.readthedocs.io"><img align="center" src="https://raw.githubusercontent.com/probabilistic-numerics/probnum/master/docs/source/assets/img/logo/probnum_logo_dark_txtbelow.svg" alt="probabilistic numerics" width="400" style="padding-right: 10px; padding left: 10px;" title="Probabilistic Numerics in Python"/>
    </a>
    <h3>Learn to Approximate. Approximate to Learn.</h3>
    <p>Probabilistic Numerics in Python.</p>
</div>

---

<div align="center">

<h4 align="center">
  <a href="https://probnum.readthedocs.io">Home</a> |
  <a href="https://probnum.readthedocs.io/en/latest/tutorials.html">Tutorials</a> |  
  <a href="https://probnum.readthedocs.io/en/latest/api.html">API Reference</a> |
  <a href="https://probnum.readthedocs.io/en/latest/development.html">Contributing</a>
</h4>

[![CI build](https://img.shields.io/github/workflow/status/probabilistic-numerics/probnum/Linting?style=flat-square&logo=github&logoColor=white&label=CI-build)](https://github.com/probabilistic-numerics/probnum/actions?query=workflow%3ACI-build)
[![Coverage Status](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?style=flat-square&label=Coverage&logo=codecov&logoColor=white)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/master)
[![Benchmarks](http://img.shields.io/badge/Benchmarks-asv-blueviolet.svg?style=flat-square&logo=swift&logoColor=white)](https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/)
[![PyPI](https://img.shields.io/pypi/v/probnum?style=flat-square&label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/probnum/)

</div>

---

**ProbNum** is a Python toolkit for solving numerical problems in linear algebra, optimization, quadrature and
differential equations. ProbNum solvers not only estimate the solution of the numerical problem, but also its uncertainty (numerical error) which arises from finite computational resources, discretization and stochastic input. This numerical uncertainty can be used in downstream decisions.

Currently, available solvers are:

- **Linear solvers:** Solve *Ax = b* for *x*.
- **ODE solvers:** Solve *&#7823;(t) = f(&#8201;y(t), t&#8201;)* for *y*.
- **Integral solvers (quadrature):** Solve *F = &#x222b; f(x)&#8201;p(x)&#8201;dx* for *F*.

Lower level structure includes:

- **Random variables and random processes**, as well as arithmetic operations thereof.
- Memory-efficient and lazy implementation of **linear operators**.
- **Filtering and smoothing** for (probabilistic) state-space models, mostly variants of Kalman filters.

ProbNum is underpinned by the research field probabilistic numerics (PN), which lies at the intersection of machine learning and numerics.
PN aims to quantify uncertainty arising from intractable or incomplete numerical computation and from stochastic input 
using the tools of probability theory. The general vision of probabilistic numerics is to provide well-calibrated 
probability measures over the output of a numerical routine, which then can be propagated along the chain of 
computation.


## Installation
To get started install ProbNum using `pip`.
```bash
pip install probnum
```
Alternatively, you can install the latest version from source.
```bash
pip install git+https://github.com/probabilistic-numerics/probnum.git
```

> Note: This package is currently work in progress, therefore interfaces are subject to change.

## Documentation and Examples
For tips on getting started and how to use this package please refer to the
[**documentation**](https://probnum.readthedocs.io). It contains a 
[quickstart guide](https://probnum.readthedocs.io/en/latest/tutorials/quickstart.html) 
and Jupyter notebooks illustrating the basic usage of the ProbNum solvers.

## Package Development
This repository is currently under development and benefits from contribution to the code, examples or documentation.
Please refer to the [contribution guidelines](https://probnum.readthedocs.io/en/latest/development.html) before
making a pull request.

A list of core contributors to ProbNum can be found
[here](https://probnum.readthedocs.io/en/latest/development.html#probnum-team).

## License and Contact
This work is released under the [MIT License](https://github.com/probabilistic-numerics/probnum/blob/master/LICENSE.txt).

Please submit an [issue on GitHub](https://github.com/probabilistic-numerics/probnum/issues/new) to report bugs or
request changes.
