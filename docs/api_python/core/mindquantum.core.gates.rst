mindquantum.core.gates
======================

.. py:module:: mindquantum.core.gates


量子门模块，提供不同的量子门。

基类
-------------

.. mscnautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.BasicGate
    mindquantum.core.gates.NoneParameterGate
    mindquantum.core.gates.ParameterGate
    mindquantum.core.gates.QuantumGate
    mindquantum.core.gates.NoiseGate

通用量子门
-------------

.. mscnmathautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.CNOTGate
    mindquantum.core.gates.FSim
    mindquantum.core.gates.GlobalPhase
    mindquantum.core.gates.HGate
    mindquantum.core.gates.IGate
    mindquantum.core.gates.ISWAPGate
    mindquantum.core.gates.Measure
    mindquantum.core.gates.PhaseShift
    mindquantum.core.gates.Rn
    mindquantum.core.gates.RX
    mindquantum.core.gates.Rxx
    mindquantum.core.gates.Rxy
    mindquantum.core.gates.Rxz
    mindquantum.core.gates.RY
    mindquantum.core.gates.Ryy
    mindquantum.core.gates.Ryz
    mindquantum.core.gates.RZ
    mindquantum.core.gates.Rzz
    mindquantum.core.gates.RotPauliString
    mindquantum.core.gates.SGate
    mindquantum.core.gates.SWAPalpha
    mindquantum.core.gates.SWAPGate
    mindquantum.core.gates.SXGate
    mindquantum.core.gates.TGate
    mindquantum.core.gates.U3
    mindquantum.core.gates.XGate
    mindquantum.core.gates.YGate
    mindquantum.core.gates.ZGate
    mindquantum.core.gates.GroupedPauli
    mindquantum.core.gates.Givens

功能量子门
-------------

.. mscnautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.UnivMathGate
    mindquantum.core.gates.gene_univ_parameterized_gate
    mindquantum.core.gates.BarrierGate

预实例化门
----------

如下量子门是预实例化的量子门，可直接作为对应量子门的实例来使用。

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - functional
     - gates
   * - mindquantum.core.gates.CNOT
     - :class:`~.core.gates.CNOTGate`
   * - mindquantum.core.gates.I
     - :class:`~.core.gates.IGate`
   * - mindquantum.core.gates.ISWAP
     - :class:`~.core.gates.ISWAPGate`
   * - mindquantum.core.gates.H
     - :class:`~.core.gates.HGate`
   * - mindquantum.core.gates.S
     - :class:`~.core.gates.PhaseShift` (numpy.pi/2)
   * - mindquantum.core.gates.SWAP
     - :class:`~.core.gates.SWAPGate`
   * - mindquantum.core.gates.SX
     - :class:`~.core.gates.SXGate`
   * - mindquantum.core.gates.T
     - :class:`~.core.gates.PhaseShift` (numpy.pi/4)
   * - mindquantum.core.gates.X
     - :class:`~.core.gates.XGate`
   * - mindquantum.core.gates.Y
     - :class:`~.core.gates.YGate`
   * - mindquantum.core.gates.Z
     - :class:`~.core.gates.ZGate`

量子信道
-------------

.. mscnmathautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.AmplitudeDampingChannel
    mindquantum.core.gates.BitFlipChannel
    mindquantum.core.gates.BitPhaseFlipChannel
    mindquantum.core.gates.DepolarizingChannel
    mindquantum.core.gates.KrausChannel
    mindquantum.core.gates.PauliChannel
    mindquantum.core.gates.GroupedPauliChannel
    mindquantum.core.gates.PhaseDampingChannel
    mindquantum.core.gates.PhaseFlipChannel
    mindquantum.core.gates.ThermalRelaxationChannel

功能类
-------------

.. mscnautosummary::
    :toctree: gates
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.MeasureResult
    mindquantum.core.gates.Power
