Class mindquantum.algorithm.nisq.Max2SATAnsatz(clauses, depth=1)

    Max-2-SAT和萨茨。欲了解更多详细信息，请参考https://arxiv.org/pdf/1906.11259.pdf.

    .. math::

        U(\beta, \gamma) = e^{-\beta_pH_b}e^{-\gamma_pH_c}
        \cdots e^{-\beta_0H_b}e^{-\gamma_0H_c}H^{\otimes n}

    在哪里，

    .. math::

        H_b = \sum_{i\in n}X_{i}, H_c = \sum_{l\in m}P(l)

    这里：数学：`n`是布尔变量的数量，数学：`m`是总子句的数量，数学：`P(l)`是一级投影仪。

    参数:
        clauses (list[tuple[int]]): Max-2-SAT结构。列表的每个元素都是一个由长度为2的元组表示的子句。
            元组的元素必须是非零整数。例如，（2,-3）代表子句：数学：`x_2\lor\lnot x_3`。
        depth (int): Max-2-SAT的深度。默认值：1。

    样例:
        >>> from mindquantum.algorithm.nisq.qaoa import Max2SATAnsatz
        >>> clauses = [(2, -3)]
        >>> max2sat = Max2SATAnsatz(clauses, 1)
        >>> max2sat.circuit
        q1: ──H─────RZ(0.5*beta_0)────●───────────────────────●────RX(alpha_0)──
                                      │                       │
        q2: ──H────RZ(-0.5*beta_0)────X────RZ(-0.5*beta_0)────X────RX(alpha_0)──

        >>> max2sat.hamiltonian
        0.25 [] +
        0.25 [Z1] +
        -0.25 [Z1 Z2] +
        -0.25 [Z2]
        >>> sats = max2sat.get_sat(4, np.array([4, 1]))
        >>> sats
        ['001', '000', '011', '010']
        >>> for i in sats:
        >>>     print(f'sat value: {max2sat.get_sat_value([i])}')
        sat value: 1
        sat value: 0
        sat value: 2
        sat value: 1
       