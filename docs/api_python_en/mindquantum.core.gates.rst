mindquantum.core.gates
======================

.. automodule:: mindquantum.core.gates

Base Class
-------------

.. autosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.BasicGate
    mindquantum.core.gates.NoneParameterGate
    mindquantum.core.gates.ParameterGate

Quantum Gate
-------------

.. msmathautosummary::
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

Functional Gate
----------------

.. autosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.UnivMathGate
    mindquantum.core.gates.gene_univ_parameterized_gate
    mindquantum.core.gates.BarrierGate

pre-instantiated gate
----------------------

The gates blow are the pre-instantiated quantum gates, which can be used directly as an instance of quantum gate.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - pre-instantiated gate
     - gate
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

Quantum Channel
----------------

.. msmathautosummary::
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

Functional Class
-----------------

.. autosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.core.gates.MeasureResult
    mindquantum.core.gates.Power
