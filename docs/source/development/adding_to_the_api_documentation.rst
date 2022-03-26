Adding to the API Documentation
===============================

Documentation is an integral part of every collaborative software
project. Good documentation not only encourages users of the package to
try out different functionalities, but it also makes maintaining and
expanding code significantly easier. Every code contribution to the
package must come with appropriate documentation of the API. This guide
details how to do this.

Docstrings
----------

The main form of documentation are docstrings, multi-line comments
beneath a class or function definition with a specific syntax, which
detail its functionality. This package uses the `NumPy docstring
format <https://numpydoc.readthedocs.io/en/latest/format.html#numpydoc-docstring-guide%3E>`__.
As a rule, all functions which are exposed to the user *must* have
appropriate docstrings. Below is an example of a docstring for a
probabilistic numerical method 
named ``problinsolve`` defined in ``_problinsolve.py`` under ``linalg`` module.

.. literalinclude:: ../../../src/probnum/linalg/_problinsolve.py
   :lines: 1-163

**General Rules**

-  Cover ``Parameters``, ``Returns``, ``Raises`` and ``Examples``, if
   applicable, in every publicly visible docstring—in that order.
-  Examples are tested via doctest. Ensure ``doctest`` does not fail by
   running the test suite.
-  Include appropriate ``References``, in particular for probabilistic
   numerical methods.
-  Do not use docstrings as a clutch for spaghetti code!

**Parameters**

-  Parameter types are automatically documented via type hints in the
   function signature.
-  Always provide shape hints for objects with a ``.shape`` attribute in
   the following form:

.. code:: python

   """
   Parameters
   ----------
   arr
       *(shape=(m, ) or (m, n))* -- Parameter array of an example function.
   """

-  Hyperparameters should have default values and explanations on how to
   choose them.
-  For callables provide the expected signature as part of the
   docstring: ``foobar(x, y, z, \*\*kwargs)``. Backslashes remove
   semantic meaning from special characters.

**Style**

-  Stick to the imperative style of writing in the docstring header
   (i.e.: first line).

   -  Yes: “Compute the value”.
   -  No: “This function computes the value / Let’s compute the value”.

   The rest of the explanation talks about the function, e. g. “This
   function computes the value by computing another value”.
-  Use full sentences inside docstrings when describing something.

   -  Yes: “This value is irrelevant, because it is not being passed on”
   -  No: “Value irrelevant, not passed on”.

-  When in doubt, more explanation rather than less. A little text
   inside an example can be helpful, too.
-  A little maths can go a long way, but too much usually adds
   confusion.

Interface Documentation
-----------------------

Which functions and classes actually show up in the documentation is
determined by an ``__all__`` statement in the corresponding
``__init__.py`` file inside a module. The order of this list is also
reflected in the documentation. For example, ``linalg`` has the
following ``__init__.py``:

.. literalinclude:: ../../../src/probnum/linalg/__init__.py


If you are documenting a subclass, which has a different path in the
file structure than the import path due to ``__all__`` statements, you
can correct the links to superclasses in the documentation via the
``.__module__`` attribute.

Sphinx
------

ProbNum uses `Sphinx <https://www.sphinx-doc.org/en/master/>`__ to parse
docstrings in the codebase automatically and to create its API
documentation. You can configure Sphinx itself or its extensions in the
``./docs/conf.py`` file.

.. code:: ipython3

    from IPython.display import Image
    
    display(Image(filename="../assets/img/developer_guides/sphinx_logo.png", embed=True))



.. image:: ../assets/img/developer_guides/sphinx_logo.png


ProbNum makes use of a number of Sphinx plugins to improve the API
documentation, for example to parse this Jupyter notebook. The full list
of used packages can be found in ``./docs/sphinx-requirements.txt`` and
``./docs/notebook-requirements.txt``.

Building and Viewing the Documentation
--------------------------------------

In order to build the documentation locally and view the HTML version of
the API documentation, simply run:

.. code:: bash

   tox -e docs

This creates a static web page under ``./docs/_build/html/`` which you
can view in your browser by opening ``./docs/_build/html/intro.html``.

Alternatively, if you want to build the docs in your current environment
you can manually execute

.. code:: bash

   cd docs
   make clean
   make html

For more information on ``tox``, check out the `general development
instructions <../development/pull_request.md>`__.
