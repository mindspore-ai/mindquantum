.. py:class:: mindquantum.core.operators.FermionOperator(term=None, coefficient=1.0)

    费米子算子，如FermionOperator('4^ 3 9 3^')表示 :math:`a_4^\dagger a_3 a_9 a_3^\dagger`。
    这些是描述费米子系统的基本运算符，如分子系统。
    FermionOperator遵循反交换关系。

    **参数：**

    - **terms** (str) - 费米子算子的输入项。默认值：None。
    - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 对应单运算符的系数。默认值：1.0。
   