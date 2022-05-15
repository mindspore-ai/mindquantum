mindquantum.core.operators.commutator(left_operator, right_operator)

    计算两个运算符的换向器。

    参数:
        left_operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            费米尔运营商或Qubit运营商。
        right_operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            费米尔运营商或Qubit运营商。

    异常:
        TypeError: operator_a and operator_b are not of the same type.

    样例:
        >>> from mindquantum.core.operators import QubitOperator,FermionOperator
        >>> from mindquantum.core.operators import commutator
        >>> qub_op1 = QubitOperator("X1 Y2")
        >>> qub_op2 = QubitOperator("X1 Z2")
        >>> commutator(qub_op1, qub_op1)
        0
        >>> commutator(qub_op1, qub_op2)
        2j [X2]
    