mindquantum.simulator.mqchem.prepare_uccsd_vqe
=================================================

.. py:function:: mindquantum.simulator.mqchem.prepare_uccsd_vqe(molecular, threshold=1e-6)

    为使用 MQChemSimulator 进行 UCCSD-VQE 模拟准备所有组件。

    该工厂函数通过以下步骤简化了VQE模拟的设置：
    1. 使用 :function:`~.algorithm.nisq.chem.uccsd_singlet_generator` 生成所有单重态UCCSD激发算符。
    2. 从 `molecular` 数据中包含的预先计算的CCSD结果中提取相应的振幅。
    3. 根据 `threshold` 筛选激发（基于它们的CCSD振幅）。
    4. 使用 :class:`~.simulator.mqchem.UCCExcitationGate` 构建参数化的UCCSD拟设线路。
    5. 创建一个 :class:`~.simulator.mqchem.CIHamiltonian` 用于期望值评估。
    6. 返回运行VQE实验所需的所有组件。

    参数：
        - **molecular** (openfermion.MolecularData) - 分子数据对象，必须包含CCSD计算结果。
        - **threshold** (float) - CCSD振幅的阈值。振幅低于此值的激发将被丢弃。默认值：``1e-6``。

    返回：
        - **hamiltonian** (mqchem.CIHamiltonian), CI空间哈密顿量。
        - **ansatz_circuit** (Circuit), 参数化的UCCSD拟设线路。
        - **initial_amplitudes** (numpy.ndarray), 与 `ansatz_circuit` 中参数对应的CCSD振幅，适合作为优化器的初始猜测。
