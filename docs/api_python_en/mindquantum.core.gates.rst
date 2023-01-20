mindquantum.core.gates
======================

.. automodule:: mindquantum.core.gates

.. currentmodule:: mindquantum.core.gates

Basic
-------------

.. autosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    BasicGate
    NoneParameterGate
    ParameterGate

Quantum Gate
-------------

.. msmathautosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    CNOTGate
    FSim
    GlobalPhase
    HGate
    IGate
    ISWAPGate
    Measure
    PhaseShift
    RX
    RY
    RZ
    SGate
    SWAPGate
    TGate
    U3
    UnivMathGate
    XGate
    XX
    YGate
    YY
    ZGate
    ZZ
    gene_univ_parameterized_gate
    BarrierGate

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

    AmplitudeDampingChannel
    BitFlipChannel
    BitPhaseFlipChannel
    DepolarizingChannel
    KrausChannel
    PauliChannel
    PhaseDampingChannel
    PhaseFlipChannel

Functional Class
-----------------

.. autosummary::
    :toctree:
    :nosignatures:
    :template: classtemplate.rst

    MeasureResult
    Power
