# <a href="https://probnum.readthedocs.io"><img align="left" src="https://raw.githubusercontent.com/probabilistic-numerics/probnum/master/docs/source/img/pn_logo.png" alt="probabilistic numerics" width="64" style="padding-right: 10px; padding left: 10px;" title="Probabilistic Numerics in Python"/></a> ProbNum
[![CI build](https://img.shields.io/github/workflow/status/probabilistic-numerics/probnum/Linting?logo=github&logoColor=white&label=CI-build)](https://github.com/probabilistic-numerics/probnum/actions?query=workflow%3ACI-build)
[![Coverage Status](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?label=Coverage&logo=codecov&logoColor=white)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/master)
[![Documentation](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Documentation)](https://probnum.readthedocs.io)
[![Tutorials](https://img.shields.io/badge/Tutorials-Jupyter-579ACA.svg?&logo=Jupyter&logoColor=white)](https://mybinder.org/v2/gh/probabilistic-numerics/probnum/master?filepath=docs%2Fsource%2Ftutorials)
[![Benchmarks](http://img.shields.io/badge/Benchmarks-asv-blueviolet.svg?style=flat&logo=swift&logoColor=white)](https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/)
[![PyPI](https://img.shields.io/pypi/v/probnum?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/probnum/)

---

ProbNum is a Python toolkit for solving numerical problems in linear algebra, optimization, quadrature and 
differential equations. ProbNum solvers not only estimate the solution of the numerical problem, but also its 
uncertainty (the error) which inevitably arises from finite computational resources, discretization, and stochastic input. 

Currently, available solvers are:

- Linear solver: Solve 
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mi>A</mi>
    <mi>x</mi>
    <mo>=</mo>
    <mi>b</mi>
  </math>
  for
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mi>x</mi>
  </math>.
  
- ODE solvers: Solve
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mrow data-mjx-texclass="ORD">
      <mover>
        <mi>y</mi>
        <mo>&#x2D9;</mo>
      </mover>
    </mrow>
    <mo stretchy="false">(</mo>
    <mi>t</mi>
    <mo stretchy="false">)</mo>
    <mo>=</mo>
    <mi>f</mi>
    <mo stretchy="false">(</mo>
    <mi>y</mi>
    <mo stretchy="false">(</mo>
    <mi>t</mi>
    <mo stretchy="false">)</mo>
    <mo>,</mo>
    <mi>t</mi>
    <mo stretchy="false">)</mo>
  </math>
  for 
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mi>y</mi>
  </math>.
  
- Quadrature solvers: Solve 
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mi>F</mi>
    <mo>=</mo>
    <msub>
      <mo data-mjx-texclass="OP">&#x222B;</mo>
      <mrow data-mjx-texclass="ORD">
        <mi mathvariant="normal">&#x3A9;</mi>
      </mrow>
    </msub>
    <mi>f</mi>
    <mo stretchy="false">(</mo>
    <mi>x</mi>
    <mo stretchy="false">)</mo>
    <mi>p</mi>
    <mo stretchy="false">(</mo>
    <mi>x</mi>
    <mo stretchy="false">)</mo>
    <mi>d</mi>
    <mi>x</mi>
  </math>  
  for 
  <math xmlns="http://www.w3.org/1998/Math/MathML">
    <mi>F</mi>
  </math>.

Lower level structure includes:

- Some efficient random variable arithmetics, especially for Gaussian distributions.
- Structure for efficient computation with linear operators.
- Structure for random processes (espcially Gauss-Markov processes).
- Filters and smothers for probabilistic state space models, mostly variants of Kalman filters.  
- Structure for probabilistic state space models.

The ProbNum library is related to [Probabilistic numerics](http://probabilistic-numerics.org/) (PN) 
which is a research field existing at the intersection of machine learning, and numerics.
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
[**documentation**](https://probnum.readthedocs.io). It contains a [quickstart guide](https://probnum.readthedocs.io/en/latest/introduction/quickstart.html) and Jupyter notebooks illustrating the basic usage of implemented probabilistic numerics routines.

## Package Development
This repository is currently under development and benefits from contribution to the code, examples or documentation.
Please refer to the [contribution guidelines](https://probnum.readthedocs.io/en/latest/development/contributing.html) before
making a pull request.

A list of core contributors to ProbNum can be found
[here](https://probnum.readthedocs.io/en/latest/development/code_contributors.html).

## License and Contact
This work is released under the [MIT License](https://github.com/probabilistic-numerics/probnum/blob/master/LICENSE.txt).

Please submit an [issue on GitHub](https://github.com/probabilistic-numerics/probnum/issues/new) to report bugs or
request changes.
