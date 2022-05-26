.. py:class:: mindquantum.algorithm.nisq.MaxCutAnsatz(graph, depth=1)

    MaxCut ansatz。欲了解更多详细信息，请访问https://arxiv.org/pdf/1411.4028.pdf.

    .. math::

    U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
    \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    .. math::

    H_b = \sum_{i\in n}X_{i}, H_c = \sum_{(i,j)\in C}Z_iZ_j

    这里： :math:`n` 是节点的集合， :math:`C` 是图的边的集合。

    **参数：**

    - **graph** (list[tuple[int]]) – 图结构。图的每个元素都是由两个节点构造的边。
    - **depth** (int) – MaxCut ansatz的深度。默认值：1。
