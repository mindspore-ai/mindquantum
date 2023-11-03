mindquantum.core.circuit
========================

.. automodule:: mindquantum.core.circuit

Class
---------------

.. autosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.circuit.Circuit
    mindquantum.core.circuit.SwapParts
    mindquantum.core.circuit.UN

Function
---------------

.. autosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.circuit.add_prefix
    mindquantum.core.circuit.add_suffix
    mindquantum.core.circuit.apply
    mindquantum.core.circuit.as_ansatz
    mindquantum.core.circuit.as_encoder
    mindquantum.core.circuit.change_param_name
    mindquantum.core.circuit.controlled
    mindquantum.core.circuit.dagger
    mindquantum.core.circuit.decompose_single_term_time_evolution
    mindquantum.core.circuit.pauli_word_to_circuits
    mindquantum.core.circuit.shift
    mindquantum.core.circuit.qfi
    mindquantum.core.circuit.partial_psi_partial_psi
    mindquantum.core.circuit.partial_psi_psi

Channel adder
-------------

.. autosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.circuit.ChannelAdderBase
    mindquantum.core.circuit.NoiseChannelAdder
    mindquantum.core.circuit.MeasureAccepter
    mindquantum.core.circuit.ReverseAdder
    mindquantum.core.circuit.NoiseExcluder
    mindquantum.core.circuit.BitFlipAdder
    mindquantum.core.circuit.MixerAdder
    mindquantum.core.circuit.SequentialAdder
    mindquantum.core.circuit.QubitNumberConstrain
    mindquantum.core.circuit.QubitIDConstrain
    mindquantum.core.circuit.GateSelector
    mindquantum.core.circuit.DepolarizingChannelAdder

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
