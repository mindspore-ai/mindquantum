.. py:class:: mindquantum.core.operators.QubitOperator(term=None, coefficient=1.0)

    作用于量子位的项的总和，例如 0.5 * 'X1 X5' + 0.3 * 'Z1 Z2'。
    项是一个作用于n个量子位的运算符，可以表示为：coefficient * local_operator[0] x ... x local_operator[n-1]，其中x是张量乘积。
    本地运算符是作用于一个量子位的Pauli运算符（'I'，'X'，'Y'或者'Z'）。
    在数学符号中，一个QubitOperator是例如0.5 * 'X1 X5'的项，它意味着Pauli X运算符作用于量子位1和5，而恒等运算符作用于其余所有量子位。

    请注意，由QubitOperator算子组成的哈密顿量应该是一个厄米特算子，因此要求所有项的系数必须是实数。

    QubitOperator的属性设置如下：operators = ('X', 'Y', 'Z')，different_indices_commute = True。

    **参数：**

    - **term** (str) - 量子位运算符的输入项。默认值：None。
    - **coefficient** (Union[numbers.Number, str, ParameterResolver]) - 此量子位运算符的系数，可以是由字符串、符号或参数解析器表示的数字或变量。默认值：1.0。
   