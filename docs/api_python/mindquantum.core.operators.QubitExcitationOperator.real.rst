mindquantum.core.operators.QubitExcitationOperator.real

        将系数转换为其实部。

        返回:
            量子位激励运算符，这个量子位激励运算符的实际部分。

        样例:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> f = QubitExcitationOperator(((1, 0),), 1 + 2j)
            >>> f += QubitExcitationOperator(((1, 1),), 'a')
            >>> f.real.compress()
            1.0 [Q1] +
            a [Q1^]
        