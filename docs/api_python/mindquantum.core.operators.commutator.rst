mindquantum.core.operators.commutator
======================================

.. py:function:: mindquantum.core.operators.commutator(left_operator, right_operator)

    计算两个算子的对易。

    参数：
        - **left_operator** (Union[FermionOperator, QubitOperator, QubitExcitationOperator]) - 第一个算子，类型是 `FermionOperator` 或者 `QubitOperator`。
        - **right_operator** (Union[FermionOperator, QubitOperator, QubitExcitationOperator]) - 第二个算子，类型是 `FermionOperator` 或者 `QubitOperator`。

    异常：
        - **TypeError** - `left_operator` 和 `right_operator` 不是相同的类型。
