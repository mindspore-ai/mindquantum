mindquantum.core.circuit
========================

.. automodule:: mindquantum.core.circuit

.. currentmodule:: mindquantum.core.circuit

Class
---------------

.. autosummary::
    :toctree:
    :nosignatures:

    Circuit
    SwapParts
    U3
    UN

Function
---------------

.. autosummary::
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

shortcut
----------

The operators blow are shortcut of correspand quantum circuit operators.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - shortcut
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
