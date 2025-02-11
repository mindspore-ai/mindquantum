mindquantum.algorithm.mapping.SABRE
===================================

.. py:class:: mindquantum.algorithm.mapping.SABRE(circuit: Circuit, topology: QubitsTopology)

    SABRE（基于SWAP的双向启发式搜索）算法用于量子比特映射优化。

    由于实际量子硬件中的物理约束，并非所有量子比特都能直接交互。SABRE 算法通过插入 SWAP 门和重新映射量子比特，
    使任意量子线路能够在特定的量子硬件拓扑结构上运行。该算法采用双向启发式搜索方法，通过考虑当前和未来的门操作
    来最小化代价函数，从而找到最优的映射方案。

    参考文献：
        Gushu Li, Yufei Ding, Yuan Xie: "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices",
        ASPLOS 2019. https://arxiv.org/abs/1809.02573

    参数：
        - **circuit** (:class:`~.core.circuit.Circuit`) - 需要映射的量子线路。当前仅支持由单量子比特门和双量子比特门（包括受控门）组成的线路。
        - **topology** (:class:`~.device.QubitsTopology`) - 硬件量子比特拓扑结构。当前仅支持连通图。

    .. py:method:: solve(iter_num: int = 5, w: float = 0.5, delta1: float = 0.3, delta2: float = 0.2)

        使用 SABRE 算法求解量子比特映射问题。

        该方法采用双向启发式搜索来寻找最优的量子比特映射方案。
        主要步骤包括：
        1. 生成随机初始映射
        2. 执行前向-后向-前向遍历以优化初始映射
        3. 使用优化后的映射执行最终前向遍历，生成带有 SWAP 门的物理线路

        参数：
            - **iter_num** (int，可选) - 前向-后向-前向遍历的迭代次数。每次迭代都从不同的初始映射开始。默认值：5。
            - **w** (float，可选) - 代价函数 H = H_current + w * H_future 中的权重参数。
              较大的 w (>0.5) 偏向未来操作，可能减少电路深度；
              较小的 w (<0.5) 偏向当前操作，可能减少总门数。
              默认值：0.5。
            - **delta1** (float，可选) - 单量子比特门的衰减参数。影响算法在单量子比特操作后如何更新衰减值。默认值：0.3。
            - **delta2** (float，可选) - 双量子比特门（CNOT，SWAP）的衰减参数。控制算法如何在空间和时间上分布 SWAP 操作。
              由于一个 SWAP 等于三个 CNOT，SWAP 操作会给衰减值增加 3*delta2。默认值：0.2。

        返回：
            - mapped_circuit (:class:`~.core.circuit.Circuit`)：添加 SWAP 门后与硬件拓扑兼容的量子线路
            - initial_mapping (List[int])：执行开始时从逻辑量子比特到物理量子比特的映射
            - final_mapping (List[int])：执行结束时从逻辑量子比特到物理量子比特的映射
