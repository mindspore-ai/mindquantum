mindquantum.core.operators.hermitian_conjugated(operator)

    返回费米子算子或量子算子的厄米共轭。

    参数:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]): 输入运算符。

    返回:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]), 输入运算符的隐士形式。

    样例:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.operators import hermitian_conjugated
        >>> q = QubitOperator('X0', {'a' : 2j})
        >>> hermitian_conjugated(q)
        -2.0*I*a [X0]
    