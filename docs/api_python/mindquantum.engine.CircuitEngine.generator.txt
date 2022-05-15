mindquantum.engine.CircuitEngine.generator(n_qubits, *args, **kwds)
量子电路寄存器。

        参数：
            n_qubits (int)：量子电路的量子位数。

        样例：
            >>> import mindquantum.core.gates as G
            >>> from mindquantum.engine import circuit_generator
            >>> @circuit_generator(2,prefix='p')
            ... def ansatz(qubits, prefix):
            ...     G.X | (qubits[0], qubits[1])
            ...     G.RX(prefix+'_0') | qubits[1]
            >>> print(ansatz)
            q0: ──●─────────────
                  │
            q1: ──X────RX(p_0)──
            >>> print(type(ansatz))
            <class 'mindquantum.core.circuit.circuit.Circuit'>
'>
