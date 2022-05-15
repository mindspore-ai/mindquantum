mindquantum.core.operators.QubitOperator.real

        将系数转换为其实部。

        返回:
            QubitOperator，这个量子位运算符的真正部分。

        样例:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.real.compress()
            1.0 [X0] +
            a [Y0]
        