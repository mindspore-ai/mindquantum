mindquantum.algorithm.nisq.MaxCutAnsatz
========================================

.. py:class:: mindquantum.algorithm.nisq.MaxCutAnsatz(graph, depth=1)

    MaxCut ansatz。了解更多详细信息，请访问 `A Quantum Approximate Optimization Algorithm <https://arxiv.org/abs/1411.4028>`_。

    .. math::

        U(\beta, \gamma) = e^{-i\beta_pH_b}e^{-i\frac{\gamma_p}{2}H_c}
        \cdots e^{-i\beta_0H_b}e^{-i\frac{\gamma_0}{2}H_c}H^{\otimes n}

    .. math::

        H_b = \sum_{i\in n}X_{i}, H_c = \sum_{(i,j)\in C}Z_iZ_j

    这里： :math:`n` 是节点的集合， :math:`C` 是图的边的集合。

    参数：
        - **graph** (list[tuple[int]]) - 图结构。图的每个元素都是由两个节点构造的边。例如，[(0, 1), (1,2)]表示一个三节点的图，且其中一条边连接节点0和节点1，另一条边连接节点1和节点2。
        - **depth** (int) - MaxCut ansatz的深度。默认值： ``1``。

    .. py:method:: get_cut_value(partition)

        获取切割方案的切割边数。切割方案是一个list数组，该list数组由两个list数组构成，每一个list数组包含切割的节点。

        参数：
            - **partition** (list) - 图形切割方案。

        返回：
            int，给定切割方案下的切割值。

    .. py:method:: get_partition(max_n, weight)

        获取MaxCut问题的切割方案。

        参数：
            - **max_n** (int) - 需要多少个切割方案。
            - **weight** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) - MaxCut ansatz的参数值。

        返回：
            list，切割方案构成的列表。

    .. py:method:: hamiltonian
        :property:

        获取MaxCut问题的哈密顿量。

        返回：
            QubitOperator，MaxCut问题的哈密顿量。
