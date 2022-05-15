mindquantum.algorithm.nisq.Transform.jordan_wigner()

        应用乔丹-威格纳变换。乔丹-威格纳变换在本地保留初始占领编号。
        它将费米子算子的公式转换为量子位算子。

        .. math::

            a^\dagger_{j}\rightarrow \sigma^{-}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i}

            a_{j}\rightarrow \sigma^{+}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i},

        其中 :math:'\sigma_{+}=\sigma^{X}+i\sigma^{Y}'和 :math:'\sigma^{X} = \sigma^{X} - i\sigma^{Y}'是Pauli自旋上升和降低算子。

        返回:
            QubitOperator，约旦_维格纳转换后的量子位运算符。
        