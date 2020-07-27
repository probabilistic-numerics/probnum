===================================
Probabilistic Numerics in Python
===================================

|Travis Status| |Coverage Status| 


`Probabilistic numerics (PN) <http://probabilistic-numerics.org/>`_ (PN) interprets classic numerical routines as 
*inference procedures* by taking a probabilistic viewpoint. This allows principled treatment of *uncertainty arising 
from finite computational resources*. The vision of probabilistic numerics is to provide well-calibrated probability 
measures over the output of a numerical routine, which then can be propagated along the chain of computation.

This repository aims to implement methods from PN in Python 3 and to provide a common interface for them. This is
currently a work in progress, therefore interfaces are subject to change.

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

   tutorials/probability
   tutorials/linear_algebra
   tutorials/differential_equations

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   public_api/prob
   public_api/linalg
   public_api/quad
   public_api/diffeq
   public_api/filtsmooth
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




.. |Travis Status| image:: https://img.shields.io/travis/probabilistic-numerics/probnum/master.svg?logo=travis%20ci&logoColor=white&label=Travis%20CI
    :target: https://travis-ci.org/probabilistic-numerics/probnum
    :alt: ProbNum's Travis CI Status

.. |Coverage Status| image:: http://codecov.io/github/probabilistic-numerics/probnum/coverage.svg?branch=master
    :target: http://codecov.io/github/probabilistic-numerics/probnum?branch=master
    :alt: ProbNum's Coverage Status