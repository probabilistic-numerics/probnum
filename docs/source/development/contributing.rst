.. _contributing:

.. mdinclude:: ../../../CONTRIBUTING.md

Contribution Tutorials
----------------------

.. toctree::
   :maxdepth: 1
   :caption: Learn how to start contributing code to ProbNum with step-by-step instructions.

   example_notebook
   documentation
   probnum_method
   optimizing_code


Benchmarking Code
-----------------

In ProbNum computational cost with regard to time and memory is measured via a benchmark suite in :code:`probnum/benchmarks`.
Benchmarks are run by `airspeed velocity <https://asv.readthedocs.io/en/stable/>`_ which tracks
performance changes over time. You can check out `ProbNum's benchmarks <coming soon>`_ or run them locally on your
machine via:

.. code-block:: bash

    cd probnum/benchmarks
    asv run
    asv publish
    asv preview

If you want to add a new benchmark follow `this tutorial <https://asv.readthedocs.io/en/stable/writing_benchmarks.html>`_.

.. mdinclude:: ../../../AUTHORS.md