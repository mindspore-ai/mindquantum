mindquantum.core.operators.QubitExcitationOperator.imag

        将系数转换为其形象部分。

        返回:
            Qubit激励运算符，此量子比特激励运算符的图像部分。

        样例:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> f = QubitExcitationOperator(((1, 0),), 1 + 2j)
            >>> f += QubitExcitationOperator(((1, 1),), 'a')
            >>> f.imag.compress()
            2.0 [Q1]
        