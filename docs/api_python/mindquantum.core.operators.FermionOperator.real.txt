mindquantum.core.operators.FermionOperator.real

        将系数转换为其实部。

        返回:
            Fermion算子，这个费米子算子的真正部分。

        样例:
            >>> from mindquantum.core.operators import FermionOperator
            >>> f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
            >>> f.real.compress()
            1.0 [0] +
            a [0^]
        