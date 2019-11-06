# Probabilistic Numerics Code Repository

[![Build Status](https://travis-ci.com/probabilistic-numerics/probnum.svg?branch=master)](https://travis-ci.com/probabilistic-numerics/probnum)
[![Coverage Status](http://codecov.io/github/probabilistic-numerics/probnum/coverage.svg?branch=master)](http://codecov.io/github/probabilistic-numerics/probnum?branch=master)
[![Documentation Status](https://readthedocs.org/probabilistic-numerics/probnum/badge/?version=master)](http://gpflow.readthedocs.io/en/master/?badge=master)


* [Introduction](#introduction)
* [Installation and Documentation](#installation)
* [Examples](#examples)
* [Contributing Code](#contributing)
* [License and Contact](#contact)

## <a name="introduction">Introduction</a>
<a href="https://github.com/probabilistic-numerics"><img align="left" src="https://github.com/probabilistic-numerics/probnum/blob/master/img/pn_logo.png" alt="probabilistic numerics" width="128"/> [Probabilistic Numerics](http://probabilistic-numerics.org/) (PN) interprets classic numerical routines as _inference procedures_ by taking a probabilistic viewpoint. This allows principled treatment of _uncertainty arising from finite computational resources_. The vision of probabilistic numerics is to provide well-calibrated probability measures over the output of a numerical routine, which then can be propagated along the chain of computation.

This repository aims to implement standard PN algorithms in Python 3 and to provide a common interface for them. This is 
currently a work in progress (see [Contributing Code](#contributing)), therefore interfaces are subject to change.

## <a name="installation">Installation and Documentation</a>
You can install this Python 3 package using `pip` (or `pip3`):
```
pip install git+https://github.com/probabilistic-numerics/probnum.git
```
Alternatively you can clone this repository with
```
git clone https://github.com/probabilistic-numerics/probnum
cd probnum
python setup.py install
```
For tips on getting started and how to use this package please refer to the 
[documentation](https://probabilistic-numerics.github.io/probnum/).

## <a name="examples">Examples</a>
In the future we will provide [Jupyter notebooks](notebooks) to illustrate basic usage examples of implemented 
probabilistic numerics routines.

## <a name="contributing">Contributing Code</a>
This repository is currently under development and benefits from contribution to the code, examples or documentation. 
Please refer to the [contribution guide](https://github.com/probabilistic-numerics/probnum/blob/master/CONTRIBUTING.md) before making a pull request.

A list of core contributors to Probnum can be found [here](https://github.com/probabilistic-numerics/probnum/blob/master/AUTHORS.md).

## <a name="license">License and Contact</a>
This work is released under the [MIT License](LICENSE.txt).

Please submit an [issue](https://github.com/probabilistic-numerics/probnum/issues/new) to report bugs or request 
changes.
