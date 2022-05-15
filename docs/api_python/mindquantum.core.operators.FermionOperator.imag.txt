mindquantum.core.operators.FermionOperator.imag

        将系数转换为其形象部分。

        返回:
            费米子算子，这个费米子算子的形象部分。

        样例:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.imag.compress()
            2.0 [0]
        