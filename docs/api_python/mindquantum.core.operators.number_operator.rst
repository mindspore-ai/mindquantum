mindquantum.core.operators.number_operator
===========================================

.. py:function:: mindquantum.core.operators.number_operator(n_modes=None, mode=None, coefficient=1.0)

    返回 `reverse_jordan_wigner` 变换的费米数运算符。

    参数：
        - **n_modes** (int) - 系统中模式的数量。默认值： ``None``。
        - **mode** (int, optional) - 返回数运算符的模式。如果是None，则返回所有点上的总数运算符。默认值： ``None``。
        - **coefficient** (float) - 项的系数。默认值： ``1.0``。

    返回：
        FermionOperator，reverse_jordan_wigner变换的费米数运算符。
