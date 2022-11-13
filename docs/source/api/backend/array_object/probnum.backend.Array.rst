Array
=====

.. currentmodule:: probnum.backend

.. autoclass:: Array

Object representing a multi-dimensional array stored on a :class:`~probnum.backend.Device` and containing elements of the same :class:`~probnum.backend.Dtype`.

Depending on the chosen backend, :class:`~probnum.backend.Array` is an alias of
:class:`numpy.ndarray`, :class:`jax.numpy.ndarray` or :class:`torch.Tensor`.
