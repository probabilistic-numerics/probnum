Array Object
============

The basic object representing a multi-dimensional array and adjacent functionality.

.. currentmodule:: probnum.backend

Functions
---------

.. autosummary::

    ~probnum.backend.isarray

Classes
-------

+----------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.Array`  | Object representing a multi-dimensional array stored on a :class:`~probnum.backend.Device` and containing elements of the same  |
|                                  | :class:`~probnum.backend.DType`.                                                                                                |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.Scalar` | Object representing a scalar with a :class:`~probnum.backend.DType`.                                                            |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.Device` | Device, such as a CPU or GPU, on which an :class:`~probnum.backend.Array` is located.                                           |
+----------------------------------+---------------------------------------------------------------------------------------------------------------------------------+


.. toctree::
    :hidden:

    array_object/probnum.backend.isarray
    array_object/probnum.backend.Array
    array_object/probnum.backend.Device
    array_object/probnum.backend.Scalar
