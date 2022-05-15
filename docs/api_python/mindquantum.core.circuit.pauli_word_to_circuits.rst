mindquantum.core.circuit.pauli_word_to_circuits(qubitops)

    将单个保利字量子位运算符转换为量子电路。

    参数:
        qubitops (QubitOperator, Hamiltonian): 单保利词量子位运算符。

    返回:
        电路，量子电路。

    异常:
        TypeError: 如果量子位数不是量子位操作符或汉密尔顿。
        ValueError: 如果量子点是汉密尔顿式的，但不是在原点模式下。
        ValueError: 如果qubitops有多个Pauliwords。

    样例:
        >>> from mindquantum.core import X
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.circuit import pauli_word_to_circuits
        >>> qubitops = QubitOperator('X0 Y1')
        >>> pauli_word_to_circuits(qubitops) + X(1, 0)
        q0: ──X────●──
                   │
        q1: ──Y────X──
       