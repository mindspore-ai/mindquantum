mindquantum.core.circuit.Circuit.hermitian()

        去找这个量子电路的隐士。

        样例:
            >>> from mindquantum.core import Circuit
            >>> from mindquantum.core import RX
            >>> circ = Circuit(RX({'a': 0.2}).on(0))
            >>> herm_circ = circ.hermitian()
            >>> herm_circ[0].coeff
            {'a': -0.2}
        