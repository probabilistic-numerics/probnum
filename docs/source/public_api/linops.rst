probnum.linops
==============

Finite-dimensional Linear Operators.

This package implements finite dimensional linear operators. It can be used to do
linear algebra with (structured) matrices without explicitly representing them in
memory. This often allows for the definition of a more efficient matrix-vector
product. Linear operators can be applied, added, multiplied, transposed, and more as
one would expect from matrix algebra.

Several algorithms in the :mod:`probnum.linalg` library are able to operate on
:class:`~probnum.linops.LinearOperator` instances.

.. automodapi:: probnum.linops
    :no-heading:
    :no-main-docstr:
