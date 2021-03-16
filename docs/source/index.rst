=======
ProbNum
=======

|CI Status| |Coverage Status| |Tutorials| |Benchmarks| |PyPI|

----

**ProbNum implements probabilistic numerical methods in Python.** Such methods solve numerical problems from linear
algebra, optimization, quadrature and differential equations using *probabilistic inference*. This approach captures
uncertainty arising from *finite computational resources* and *stochastic input*.

----

`Probabilistic numerics <http://probabilistic-numerics.org/>`_ (PN) aims to quantify uncertainty arising from
intractable or incomplete numerical computation and from stochastic input using the tools of probability theory. The
vision of probabilistic numerics is to provide well-calibrated probability measures over the output of a numerical
routine, which then can be propagated along the chain of computation.

To get started install ProbNum using :code:`pip`.

.. code-block:: shell

   pip install probnum


Alternatively, you can install the package from source.

.. code-block:: shell

   pip install git+https://github.com/probabilistic-numerics/probnum.git


To learn how to use ProbNum check out the `quickstart guide <introduction/quickstart.html>`_ and the tutorials.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   introduction/probabilistic_numerics
   introduction/quickstart

.. toctree::
   :maxdepth: 1
   :caption: Tutorials and Examples

   tutorials/pn_methods
   tutorials/probability
   tutorials/linear_algebra
   tutorials/ordinary_differential_equations
   tutorials/bayesian_filtering_smoothing

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   public_api/diffeq
   public_api/filtsmooth
   public_api/kernels
   public_api/linalg
   public_api/linops
   public_api/problems
   public_api/quad
   public_api/randvars
   public_api/statespace
   public_api/utils

.. toctree::
   :maxdepth: 1
   :caption: Contributing to ProbNum

   development/contributing
   development/developer_guides
   development/styleguide
   development/benchmarking

.. toctree::
   :maxdepth: 1
   :caption: Other

   GitHub Repository <https://github.com/probabilistic-numerics/probnum>
   development/code_contributors
   license


Indices
"""""""

* :ref:`genindex`
* :ref:`modindex`




.. |CI Status| image:: https://img.shields.io/github/workflow/status/probabilistic-numerics/probnum/Linting?logo=github&logoColor=white&label=CI-build
    :target: https://github.com/probabilistic-numerics/probnum/actions?query=workflow%3ACI-build
    :alt: ProbNum's CI Build Status

.. |Coverage Status| image:: https://img.shields.io/codecov/c/gh/probabilistic-numerics/probnum/master?label=Coverage&logo=codecov&logoColor=white
    :target: https://codecov.io/gh/probabilistic-numerics/probnum/branch/master
    :alt: ProbNum's Coverage Status

.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Jupyter-579ACA.svg?&logo=Jupyter&logoColor=white
    :target: https://mybinder.org/v2/gh/probabilistic-numerics/probnum/master?filepath=docs%2Fsource%2Ftutorials
    :alt: ProbNum's Tutorials

.. |Benchmarks| image:: http://img.shields.io/badge/Benchmarks-asv-blueviolet.svg?style=flat&logo=swift&logoColor=white
    :target: https://probabilistic-numerics.github.io/probnum-benchmarks/benchmarks/
    :alt: ProbNum's Benchmarks

.. |PyPI| image:: https://img.shields.io/pypi/v/probnum?label=PyPI&logo=pypi&logoColor=white
    :target: https://pypi.org/project/probnum/
    :alt: ProbNum on PyPI

.. |GitHub| image:: https://logodix.com/logo/64439.png
