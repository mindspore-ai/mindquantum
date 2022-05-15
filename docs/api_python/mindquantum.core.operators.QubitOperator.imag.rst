mindquantum.core.operators.QubitOperator.imag

        将系数转换为其形象部分。

        返回:
            QubitOperator，此量子位运算符的形象部分。

        样例:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.imag.compress()
            2.0 [X0]
        