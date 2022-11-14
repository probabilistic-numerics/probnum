Scalar
======

.. currentmodule:: probnum.backend

.. autoclass:: Scalar

Object representing a scalar with a :class:`~probnum.backend.DType`.

Depending on the chosen backend :class:`~probnum.backend.Scalar` is an alias of
:class:`numpy.generic`, :class:`jax.numpy.ndarray` or :class:`torch.Tensor`.
