mindquantum.simulator.mqchem.CIHamiltonian
==========================================

.. py:class:: mindquantum.simulator.mqchem.CIHamiltonian(fermion_hamiltonian)

    一个费米子哈密顿量的包装器，用于与 :class:`~.simulator.mqchem.MQChemSimulator` 一同使用。

    该类存储一个费米子哈密顿量，以便在特定的CI空间内高效地计算期望值。

    .. note::
        此哈密顿量对象专为 `MQChemSimulator` 设计，与标准的态矢量 `Simulator` 不兼容。

    参数：
        - **fermion_hamiltonian** (FermionOperator) - 一个正规序的费米子哈密顿量。

    .. py:method:: get_cpp_obj(backend, n_qubits, n_electrons)

        返回用于模拟的C++对象。

        .. note::
            此方法供 :class:`~.simulator.mqchem.MQChemSimulator` 内部使用。

        参数：
            - **backend** (``_mq_chem.float`` 或 ``_mq_chem.double``) - C++后端模块。
            - **n_qubits** (int) - 系统中的总量子比特数（自旋轨道数）。
            - **n_electrons** (int) - 系统中的总电子数。

        返回：
            C++对象，底层绑定类型为 ``CppCIHamiltonian``，用于模拟。
