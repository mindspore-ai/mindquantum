mindquantum.dtype
=================

.. automodule:: mindquantum.dtype

支持的数据类型
-------------------

如下类型是 MindQuantum 在进行量子模拟时支持的类型。

.. list-table::
   :widths: 50 50

   * - mindquantum.float32
     - single precision real number type
   * - mindquantum.float64
     - double precision real number type
   * - mindquantum.complex64
     - single precision complex number type
   * - mindquantum.complex128
     - double precision complex number type

Function
---------------

.. autosummary::
    :toctree: dtype
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.dtype.is_double_precision
    mindquantum.dtype.is_single_precision
    mindquantum.dtype.is_same_precision
    mindquantum.dtype.precision_str
    mindquantum.dtype.to_real_type
    mindquantum.dtype.to_complex_type
    mindquantum.dtype.to_single_precision
    mindquantum.dtype.to_double_precision
    mindquantum.dtype.to_precision_like
    mindquantum.dtype.to_mq_type
    mindquantum.dtype.to_np_type
