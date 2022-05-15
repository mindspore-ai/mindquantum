Class mindquantum.core.operators.QubitOperator(term=None, coefficient=1.0)

    作用于量子位的项的总和，例如 0.5 * 'X1 X5' + 0.3 * 'Z1 Z2'。
    术语是作用于n个量子位的运算符，可以表示为：coefficient * local_operator[0] x ... x local_operator[n-1]，其中x是张量乘积。
    本地运算符是作用于一个量子位的Pauli运算符 ('I', 'X', 'Y', or 'Z')。
    在数学符号中，QubitOperator术语例如是0.5 * 'X1 X5'，这意味着Pauli X运算符作用于量子位1和5，而身份运算符作用于所有其余量子位。

    请注意，由Qubit算子组成的哈密顿量应该是一个隐数算子，因此要求所有项的系数必须是实数。

    QubitOperator的属性设置如下：operators = ('X', 'Y', 'Z')，different_indices_commute = True.

    参数:
        term (str): 量子位运算符的输入术语。默认值：None。
        coefficient (Union[numbers.Number, str, ParameterResolver]): 此量子位运算符的系数，可以是由字符串、符号或参数解析器表示的数字或变量。默认值：1.0。

    样例:
        >>> from mindquantum.core.operators import QubitOperator
        >>> ham = ((QubitOperator('X0 Y3', 0.5)
        ...         + 0.6 * QubitOperator('X0 Y3')))
        >>> ham2 = QubitOperator('X0 Y3', 0.5)
        >>> ham2 += 0.6 * QubitOperator('X0 Y3')
        >>> ham2
        1.1 [X0 Y3]
        >>> ham3 = QubitOperator('')
        >>> ham3
        1.0 []
        >>> ham_para = QubitOperator('X0 Y3', 'x')
        >>> ham_para
        x [X0 Y3]
        >>> ham_para.subs({'x':1.2})
        1.2 [X0 Y3]
    