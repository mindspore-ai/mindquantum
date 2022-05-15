mindquantum.core.operators.QubitExcitationOperator.normal_ordered()
返回Qubit激励运算符的正常有序形式。

        返回:
            QubitExcationOperator，正常有序运算符。

        样例:
            >>> from mindquantum.core.operators import QubitExcitationOperator
            >>> op = QubitExcitationOperator("7 1^")
            >>> op
            1.0 [Q7 Q1^]
            >>> op.normal_ordered()
            1.0 [Q1^ Q7]

        注:
            与费米子激励运算符不同，当交换阶数时，Qubit激励运算符不会乘以-1。
        