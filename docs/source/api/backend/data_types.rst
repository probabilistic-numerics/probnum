Data Types
==========

Fundamental (array) data types.

.. currentmodule:: probnum.backend

Functions
---------

.. autosummary::

    ~probnum.backend.asdtype
    ~probnum.backend.can_cast
    ~probnum.backend.cast
    ~probnum.backend.finfo
    ~probnum.backend.iinfo
    ~probnum.backend.is_floating_dtype
    ~probnum.backend.promote_types
    ~probnum.backend.result_type


Classes
-------

+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.DType`      | Data type of an :class:`~probnum.backend.Array`.                                                                        |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.bool`       | Boolean (``True`` or ``False``).                                                                                        |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.int32`      | A 32-bit signed integer.                                                                                                |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.int64`      | A 64-bit signed integer.                                                                                                |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.float16`    | IEEE 754 half-precision (16-bit) binary floating-point number.                                                          |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.float32`    | IEEE 754 single-precision (32-bit) binary floating-point number.                                                        |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.float64`    | IEEE 754 double-precision (64-bit) binary floating-point number.                                                        |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.complex64`  | Single-precision complex number represented by two :class:`~probnum.backend.float32`\s (real and imaginary components). |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| :class:`~probnum.backend.complex128` | Double-precision complex number represented by two :class:`~probnum.backend.float64`\s (real and imaginary components). |
+--------------------------------------+-------------------------------------------------------------------------------------------------------------------------+


.. toctree::
    :hidden:

    data_types/probnum.backend.DType
    data_types/probnum.backend.bool
    data_types/probnum.backend.int32
    data_types/probnum.backend.int64
    data_types/probnum.backend.float16
    data_types/probnum.backend.float32
    data_types/probnum.backend.float64
    data_types/probnum.backend.complex64
    data_types/probnum.backend.complex128
    data_types/probnum.backend.MachineLimitsFloatingPoint
    data_types/probnum.backend.MachineLimitsInteger
    data_types/probnum.backend.asdtype
    data_types/probnum.backend.can_cast
    data_types/probnum.backend.cast
    data_types/probnum.backend.finfo
    data_types/probnum.backend.iinfo
    data_types/probnum.backend.is_floating_dtype
    data_types/probnum.backend.promote_types
    data_types/probnum.backend.result_type
