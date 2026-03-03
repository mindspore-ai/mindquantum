mindquantum.simulator.mqchem
============================

.. py:module:: mindquantum.simulator.mqchem


基于组态相互作用（Configuration Interaction, CI）的 MindQuantum 量子化学模拟器。

该模块提供了一个专用模拟器（`MQChemSimulator`）及其配套组件（`UCCExcitationGate`、`CIHamiltonian`），用于在给定电子数子空间内高效执行量子化学模拟。
同时还提供高级工具 `prepare_uccsd_vqe`，以简化 VQE 计算流程的搭建。

Class
------

.. mscnautosummary::
    :toctree: .
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.simulator.mqchem.MQChemSimulator
    mindquantum.simulator.mqchem.CIHamiltonian
    mindquantum.simulator.mqchem.UCCExcitationGate
    mindquantum.simulator.mqchem.SequentialUCCAnsatz

Function
---------

.. mscnautosummary::
    :toctree: .
    :nosignatures:
    :template: classtemplate.rst

    mindquantum.simulator.mqchem.prepare_uccsd_vqe
