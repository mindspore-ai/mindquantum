Class mindquantum.core.operators.FermionOperator(term=None, coefficient=1.0)

    费米子算子，如费米子算子（'4^ 3 9 3^'）用于表示 :math:`a_4^\dagger a_3 a_9 a_3^\dagger`.
    这些是描述费米子系统的基本运算符，如分子系统。
    费米昂操作者遵循反换位关系。

    参数:
        terms (str): 费米子算子的输入项。默认值：None。
        coefficient (Union[numbers.Number, str, ParameterResolver]): 对应单运算符的系数默认值：1.0。

    样例:
        >>> from mindquantum.core.operators import FermionOperator
        >>> a_p_dagger = FermionOperator('1^')
        >>> a_p_dagger
        1.0 [1^]
        >>> a_q = FermionOperator('0')
        >>> a_q
        1.0 [0]
        >>> zero = FermionOperator()
        >>> zero
        0
        >>> identity= FermionOperator('')
        >>> identity
        1.0 []
        >>> para_op = FermionOperator('0 1^', 'x')
        >>> para_op
        x [0 1^]
        >>> para_dt = {'x':2}
        >>> op = para_op.subs(para_dt)
        >>> op
        2 [0 1^]
    