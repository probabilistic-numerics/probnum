# Probabilistic Numerics in Python


[![Build Status](https://img.shields.io/travis/probabilistic-numerics/probnum/master.svg?logo=travis%20ci&logoColor=white&label=Travis%20CI)](https://travis-ci.org/probabilistic-numerics/probnum)
[![Coverage Status](https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?label=Coverage&logo=codecov&logoColor=white)](https://codecov.io/gh/probabilistic-numerics/probnum/branch/master)
[![Documentation](https://img.shields.io/readthedocs/probnum.svg?logo=read%20the%20docs&logoColor=white&label=Documentation)](https://probnum.readthedocs.io)
[![Benchmarks](http://img.shields.io/badge/Benchmarks-asv-blueviolet.svg?style=flat&logo=swift&logoColor=white)](https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/)
[![PyPI](https://img.shields.io/pypi/v/probnum?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/probnum/)
<br>

<a href="https://probnum.readthedocs.io"><img align="left" src="https://raw.githubusercontent.com/probabilistic-numerics/probnum/master/docs/source/img/pn_logo.png" alt="probabilistic numerics" width="128" style="padding-right: 10px; padding left: 10px;" title="Probabilistic Numerics on GitHub"/></a>
[Probabilistic Numerics](https://doi.org/10.1098/rspa.2015.0142) (PN) interprets classic numerical routines as
_inference procedures_ by taking a probabilistic viewpoint. This allows principled treatment of _uncertainty arising
from finite computational resources_. The vision of probabilistic numerics is to provide well-calibrated probability
measures over the output of a numerical routine, which then can be propagated along the chain of computation.

This repository aims to implement methods from PN in Python 3 and to provide a common interface for them. This is
currently a work in progress, therefore interfaces are subject to change.

## Installation
To get started install ProbNum using `pip`.
```bash
pip install probnum
```
Alternatively, you can install the latest version from source.
```bash
pip install git+https://github.com/probabilistic-numerics/probnum.git
```

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
