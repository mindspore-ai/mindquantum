Class mindquantum.core.operators.Hamiltonian(hamiltonian)

    QubitOperator汉密尔顿包装器。

    参数:
        hamiltonian (QubitOperator): 保利词量子比特运算符。

    样例:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum import Hamiltonian
        >>> ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
    