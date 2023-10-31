mindquantum.dtype
=================

.. py:module:: mindquantum.dtype


MindQuantum 数据类型模拟。

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

Memory consuming
-------------------

下表展示全振幅量子态内存占用与比特数的关系:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - qubit number
     - complex128
     - complex64
   * - 6
     - 1kB
     - 0.5kB
   * - 16
     - 1MB
     - 0.5MB
   * - 26
     - 1GB
     - 0.5GB
   * - 30
     - 16GB
     - 8GB
   * - 36
     - 1TB
     - 0.5TB
   * - 40
     - 16TB
     - 8TB
   * - 46
     - 1PB
     - 0.5PB

Function
---------------

.. mscnautosummary::
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
