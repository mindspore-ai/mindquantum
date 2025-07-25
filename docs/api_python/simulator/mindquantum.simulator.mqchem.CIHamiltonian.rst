mindquantum.simulator.mqchem.CIHamiltonian
==========================================

.. py:class:: mindquantum.simulator.mqchem.CIHamiltonian(fermion_hamiltonian)

    一个费米子哈密顿量的包装器，用于与 :class:`~.simulator.mqchem.MQChemSimulator` 一同使用。

    该类存储一个费米子哈密顿量，以便在特定的CI空间内高效地计算期望值。

    .. note::
        此哈密顿量对象专为 `MQChemSimulator` 设计，与标准的态矢量 `Simulator` 不兼容。

    参数：
        - **fermion_hamiltonian** (FermionOperator) - 一个正规序的费米子哈密顿量。
