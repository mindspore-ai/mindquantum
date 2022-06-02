.. py:class:: mindquantum.algorithm.nisq.Max2SATAnsatz(clauses, depth=1)

    Max-2-SAT ansatz。了解更多详细信息，请参考https://arxiv.org/pdf/1906.11259.pdf。

    .. math::

    U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
    \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    .. math::

    H_b = \sum_{i\in n}X_{i}, H_c = \sum_{l\in m}P(l)

    :math:`n` 是布尔变量的数量， :math:`m` 是总子句的数量， :math:`P(l)` 是第一级投影。

    **参数：**

    - **clauses** (list[tuple[int]]) - Max-2-SAT结构。列表的每个元素都是一个由长度为2的元组表示的子句。元组的元素必须是非零整数。例如，（2,-3）代表子句： :math:`x_2\lor\lnot x_3`。
    - **depth** (int) - Max-2-SAT的深度。默认值：1。
