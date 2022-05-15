Class mindquantum.algorithm.nisq.MaxCutAnsatz(graph, depth=1)

    MaxCut安萨兹。欲了解更多详细信息，请访问https://arxiv.org/pdf/1411.4028.pdf.

    .. math::

        U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
        \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    在哪里，

    .. math::

        H_b = \sum_{i\in n}X_{i}, H_c = \sum_{(i,j)\in C}Z_iZ_j

    这里：数学：`n`是节点集，数学：`C`是图的边集。

    参数:
        graph (list[tuple[int]]): 图结构。图的每个元素都是由两个节点构造的边。
        depth (int): 最大切割安萨茨的深度。默认值：1。

    样例:
        >>> from mindquantum.algorithm.nisq.qaoa import MaxCutAnsatz
        >>> graph = [(0, 1), (1, 2), (0, 2)]
        >>> maxcut = MaxCutAnsatz(graph, 1)
        >>> maxcut.circuit
        q0: ──H────ZZ(beta_0)──────────────────ZZ(beta_0)────RX(alpha_0)──
                       │                           │
        q1: ──H────ZZ(beta_0)────ZZ(beta_0)────────┼─────────RX(alpha_0)──
                                     │             │
        q2: ──H──────────────────ZZ(beta_0)────ZZ(beta_0)────RX(alpha_0)──

        >>> maxcut.hamiltonian
        1.5 [] +
        -0.5 [Z0 Z1] +
        -0.5 [Z0 Z2] +
        -0.5 [Z1 Z2]
        >>> maxcut.hamiltonian
        >>> partitions = maxcut.get_partition(5, np.array([4, 1]))
        >>> for i in partitions:
        >>>     print(f'partition: left: {i[0]}, right: {i[1]}, cut value: {maxcut.get_cut_value(i)}')
        partition: left: [2], right: [0, 1], cut value: 2
        partition: left: [0, 1], right: [2], cut value: 2
        partition: left: [0], right: [1, 2], cut value: 2
        partition: left: [0, 1, 2], right: [], cut value: 0
        partition: left: [], right: [0, 1, 2], cut value: 0
       