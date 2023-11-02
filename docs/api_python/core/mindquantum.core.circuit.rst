mindquantum.core.circuit
========================

.. py:module:: mindquantum.core.circuit


量子线路模块，通过有序地组织各种量子门，我们可以轻松地搭建出符合要求的量子线路，包括参数化量子线路。本模块还包含各种预设的量子线路以及对量子线路进行高效操作的模块。

Class
---------------

.. mscnautosummary::
    :toctree: circuit
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.circuit.Circuit
    mindquantum.core.circuit.SwapParts
    mindquantum.core.circuit.UN

Function
---------------

.. mscnautosummary::
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

.. mscnautosummary::
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

functional
----------

如下的操作符是对应量子线路操作符的简写。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - high level circuit operators
   * - mindquantum.core.circuit.C
     - :class:`~.core.circuit.controlled`
   * - mindquantum.core.circuit.D
     - :class:`~.core.circuit.dagger`
   * - mindquantum.core.circuit.A
     - :class:`~.core.circuit.apply`
   * - mindquantum.core.circuit.AP
     - :class:`~.core.circuit.add_prefix`
   * - mindquantum.core.circuit.CPN
     - :class:`~.core.circuit.change_param_name`
