mindquantum.algorithm.nisq.Max2SATAnsatz
=========================================

.. py:class:: mindquantum.algorithm.nisq.Max2SATAnsatz(clauses, depth=1)

    Max-2-SAT ansatz。了解更多详细信息，请参考 `Reachability Deficits in Quantum Approximate Optimization <https://arxiv.org/abs/1906.11259>`_。

    .. math::

        U(\beta, \gamma) = e^{-i\beta_pH_b}e^{-i\frac{\gamma_p}{2}H_c}
        \cdots e^{-i\beta_0H_b}e^{-i\frac{\gamma_0}{2}H_c}H^{\otimes n}

    .. math::

        H_b = \sum_{i\in n}X_{i}, H_c = \sum_{l\in m}P(l)

    :math:`n` 是布尔变量的数量， :math:`m` 是总子句的数量， :math:`P(l)` 是第一级投影。

    参数：
        - **clauses** (list[tuple[int]]) - Max-2-SAT结构。列表的每个元素都是一个由长度为2的元组表示的子句。元组的元素必须是非零整数。例如，（2,-3）代表子句： :math:`x_2\lor\lnot x_3`。
        - **depth** (int) - Max-2-SAT的深度。默认值： ``1``。

    .. py:method:: get_sat(max_n, weight)

        获取Max-2-SAT问题的字符串。

        参数：
            - **max_n** (int) - 需要的字符串数量。
            - **weight** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) - Max-2-SAT Ansatz的参数值。

        返回：
            list，字符串列表。

    .. py:method:: get_sat_value(string)

        获取给定字符串的 `sat` 值。
        字符串是满足给定Max-2-SAT问题的所有子句的str。

        参数：
            - **string** (str) - Max-2-SAT问题的字符串。

        返回：
            int，给定字符串下的sat值。

    .. py:method:: hamiltonian
        :property:

        获取Max-2-SAT问题的哈密顿量。

        返回：
            QubitOperator，Max-2-SAT问题的哈密顿量。
