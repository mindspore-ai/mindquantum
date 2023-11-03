mindquantum.dtype
=================

.. automodule:: mindquantum.dtype

Supported data type
-------------------

The data type below is supported by MindQuantum when doing simulation.

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

The memory usage for full state vector increased with qubit number is shown as below:

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
