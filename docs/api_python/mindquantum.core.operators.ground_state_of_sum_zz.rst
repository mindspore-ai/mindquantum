mindquantum.core.operators.ground_state_of_sum_zz
=================================================

.. py:function:: mindquantum.core.operators.ground_state_of_sum_zz(ops: QubitOperator, sim='mqvector')

    计算只有泡利 :math:`Z` 项的哈密顿量的基态能量。

    参数：
        - **ops** (QubitOperator) - 只有泡利 :math:`Z` 项的哈密顿量。
        - **sim** (str) - 用什么模拟器去计算。默认值： ``'mqvector'``。

    返回：
       float，给定哈密顿量的基态能量。
