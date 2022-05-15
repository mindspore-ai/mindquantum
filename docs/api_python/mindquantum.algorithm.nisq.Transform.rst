Class mindquantum.algorithm.nisq.Transform(operator, n_qubits=None)

    费米子和量子位算子变换的类。
    方法约旦_维格纳、奇偶校验、布拉维伊_基塔耶夫、布拉维伊_基塔耶夫_树、布拉维伊_基塔耶夫_超快地将费米子算子转换为量子比特算子，它们由费米子算子初始化，返回QubitOperator。
    注意方法反转_jordan_wigner将量子位运算符转换为费米子运算符，它由QubitOperator初始化，返回费米子运算符。

    参数:
        operator (Union[FermionOperator, QubitOperator]): 需要进行转换的输入FermionOperator或QubitOperator。
        n_qubits (int): 此运算符的总量子位。默认值：None。

    样例:
        >>> from mindquantum.core.operators import FermionOperator
        >>> op1 = FermionOperator('1^')
        >>> op1
        1.0 [1^]
        >>> from mindquantum.algorithm.nisq.chem import Transform
        >>> op_transform = Transform(op1)
        >>> op_transform.jordan_wigner()
        0.5 [Z0 X1] +
        -0.5j [Z0 Y1]
        >>> op_transform.parity()
        0.5 [Z0 X1] +
        -0.5j [Y1]
        >>> op_transform.bravyi_kitaev()
        0.5 [Z0 X1] +
        -0.5j [Y1]
        >>> op_transform.ternary_tree()
        0.5 [X0 Z1] +
        -0.5j [Y0 X2]
        >>> op2 = FermionOperator('1^', 'a')
        >>> Transform(op2).jordan_wigner()
        0.5*a [Z0 X1] +
        -0.5*I*a [Z0 Y1]
    