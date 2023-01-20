mindquantum.core.circuit
========================

.. py:module:: mindquantum.core.circuit


量子线路模块，通过有序地组织各种量子门，我们可以轻松地搭建出符合要求的量子线路，包括参数化量子线路。本模块还包含各种预设的量子线路以及对量子线路进行高效操作的模块。

.. currentmodule:: mindquantum.core.circuit

Class
---------------

.. mscnautosummary::
    :toctree:
    :nosignatures:

    Circuit
    SwapParts
    U3
    UN

Function
---------------

.. mscnautosummary::
    :toctree:
    :nosignatures:

    add_prefix
    add_suffix
    apply
    as_ansatz
    as_encoder
    change_param_name
    controlled
    dagger
    decompose_single_term_time_evolution
    pauli_word_to_circuits
    shift
    qfi
    partial_psi_partial_psi
    partial_psi_psi

functional
----------

如下的操作符是对应量子线路操作符的简写。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - high level circuit operators
   * - mindquantum.core.circuit.C
     - :class:`mindquantum.core.circuit.controlled`
   * - mindquantum.core.circuit.D
     - :class:`mindquantum.core.circuit.dagger`
   * - mindquantum.core.circuit.A
     - :class:`mindquantum.core.circuit.apply`
   * - mindquantum.core.circuit.AP
     - :class:`mindquantum.core.circuit.add_prefix`
   * - mindquantum.core.circuit.CPN
     - :class:`mindquantum.core.circuit.change_param_name`
