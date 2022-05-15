mindquantum.core.operators.QubitExcitationOperator.to_qubit_operator()

        将Qubit激励运算符转换为等效Qubit运算符。

        返回:
            QubitOperator，根据Qubit激励运算符的定义，对应的QubitOperator。

        样例:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> op = QubitExcitationOperator("7^ 1")
            >>> op.to_qubit_operator()
            0.25 [X1 X7] +
            -0.25j [X1 Y7] +
            0.25j [Y1 X7] +
            (0.25+0j) [Y1 Y7]
        