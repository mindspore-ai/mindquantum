mindquantum.algorithm.mapping.MQSABRE
=====================================

.. py:class:: mindquantum.algorithm.mapping.MQSABRE(circuit: Circuit, topology: QubitsTopology, cnoterrorandlength: List[Tuple[Tuple[int, int], List[float]]])

    MQSABRE 算法用于硬件感知的量子比特映射优化。

    MQSABRE 通过将硬件特性纳入映射优化过程，扩展了 SABRE（基于SWAP的双向启发式搜索）算法。该算法分三个阶段执行初始映射和路由优化：

    1. 初始映射：使用基于图中心的方法生成初始映射，最小化频繁交互量子比特之间的平均距离。
    2. 映射优化：采用具有硬件感知代价函数的双向启发式搜索。
    3. 电路转换：插入 SWAP 门并转换电路以满足硬件约束。

    该算法使用一个加权代价函数，结合三个指标：
    H = α₁D + α₂K + α₃T
    其中：

    - D：耦合图中量子比特之间的最短路径距离
    - K：由 CNOT 和 SWAP 成功率导出的错误率指标
    - T：考虑 CNOT 和 SWAP 持续时间的门执行时间指标
    - α₁, α₂, α₃：用于平衡不同优化目标的权重参数

    参数：
        - **circuit** (:class:`~.core.circuit.Circuit`) - 需要映射的量子线路。当前仅支持由单量子比特门和双量子比特门（包括受控门）组成的线路。
        - **topology** (:class:`~.device.QubitsTopology`) - 硬件量子比特拓扑结构。必须是连通的耦合图。
        - **cnoterrorandlength** (List[Tuple[Tuple[int, int], List[float]]]) - 硬件特定的 CNOT 特性。
          每个条目包含一个元组 (i, j) 指定拓扑中的物理量子比特对，和一个列表 [error_rate, gate_time]，
          其中 error_rate 是量子比特 i 和 j 之间的 CNOT 错误率（范围：[0, 1]），gate_time 是 CNOT 执行时间（任意单位）。

    异常：
        - **ValueError** - 如果拓扑不是连通图。

    .. py:method:: solve(w: float = 0.5, alpha1: float = 0.3, alpha2: float = 0.2, alpha3: float = 0.1)

        使用 MQSABRE 算法求解量子比特映射问题。

        该方法执行三个主要步骤：
        1. 使用 Floyd-Warshall 算法构建距离矩阵 D
        2. 计算硬件特定的矩阵 K（错误率）和 T（门时间）
        3. 执行启发式搜索以优化映射，同时考虑组合代价函数

        参数：
            - **w** (float，可选) - 前瞻权重参数，用于在启发式搜索中平衡当前和未来门操作。范围：[0, 1]。
              当 w > 0.5 时，偏向未来操作，可能减少电路深度；
              当 w < 0.5 时，优先考虑当前操作，可能减少总门数。
              默认值：0.5。
            - **alpha1** (float，可选) - 代价函数中距离度量（D）的权重。较高的值优先考虑最小化量子比特距离。默认值：0.3。
            - **alpha2** (float，可选) - 错误率度量（K）的权重。较高的值优先考虑错误率较低的连接。默认值：0.2。
            - **alpha3** (float，可选) - 门时间度量（T）的权重。较高的值优先考虑更快的门执行路径。默认值：0.1。

        返回：
            - mapped_circuit (:class:`~.core.circuit.Circuit`)：插入 SWAP 门后的转换电路
            - initial_mapping (List[int])：从逻辑量子比特到物理量子比特的初始映射
            - final_mapping (List[int])：从逻辑量子比特到物理量子比特的最终映射
