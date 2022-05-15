mindquantum.core.operators.FermionOperator.normal_ordered()
返回费米子算子的正常有序形式。

        返回:
            费米子算子，正常有序的费米子算子。

        样例:
            >>> from mindquantum.core.operators import FermionOperator
            >>> origin = FermionOperator('0 1^')
            >>> origin
            1.0 [0 1^]
            >>> origin.normal_ordered()
            -1.0 [1^ 0]

        