mindquantum.core.gates
======================

.. py:module:: mindquantum.core.gates


量子门模块，提供不同的量子门。

基类
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.BasicGate
    mindquantum.core.gates.NoneParameterGate
    mindquantum.core.gates.ParameterGate

通用量子门
-------------

.. mscnmathautosummary::
    :toctree:
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
    mindquantum.core.gates.RX
    mindquantum.core.gates.Rxx
    mindquantum.core.gates.Rxy
    mindquantum.core.gates.Rxz
    mindquantum.core.gates.RY
    mindquantum.core.gates.Ryy
    mindquantum.core.gates.Ryz
    mindquantum.core.gates.RZ
    mindquantum.core.gates.Rzz
    mindquantum.core.gates.SGate
    mindquantum.core.gates.SWAPGate
    mindquantum.core.gates.TGate
    mindquantum.core.gates.U3
    mindquantum.core.gates.XGate
    mindquantum.core.gates.XX
    mindquantum.core.gates.YGate
    mindquantum.core.gates.YY
    mindquantum.core.gates.ZGate
    mindquantum.core.gates.ZZ

功能量子门
-------------

.. mscnautosummary::
    :toctree:
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
     - :class:`mindquantum.core.gates.CNOTGate`
   * - mindquantum.core.gates.I
     - :class:`mindquantum.core.gates.IGate`
   * - mindquantum.core.gates.ISWAP
     - :class:`mindquantum.core.gates.ISWAPGate`
   * - mindquantum.core.gates.H
     - :class:`mindquantum.core.gates.HGate`
   * - mindquantum.core.gates.S
     - :class:`mindquantum.core.gates.PhaseShift` (numpy.pi/2)
   * - mindquantum.core.gates.SWAP
     - :class:`mindquantum.core.gates.SWAPGate`
   * - mindquantum.core.gates.T
     - :class:`mindquantum.core.gates.PhaseShift` (numpy.pi/4)
   * - mindquantum.core.gates.X
     - :class:`mindquantum.core.gates.XGate`
   * - mindquantum.core.gates.Y
     - :class:`mindquantum.core.gates.YGate`
   * - mindquantum.core.gates.Z
     - :class:`mindquantum.core.gates.ZGate`

量子信道
-------------

.. mscnmathautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.AmplitudeDampingChannel
    mindquantum.core.gates.BitFlipChannel
    mindquantum.core.gates.BitPhaseFlipChannel
    mindquantum.core.gates.DepolarizingChannel
    mindquantum.core.gates.KrausChannel
    mindquantum.core.gates.PauliChannel
    mindquantum.core.gates.PhaseDampingChannel
    mindquantum.core.gates.PhaseFlipChannel

功能类
-------------

.. mscnautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.MeasureResult
    mindquantum.core.gates.Power
