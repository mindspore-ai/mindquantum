mindquantum.core.circuit.Circuit.reverse_qubits()

        将电路翻转到大端。

        样例:
            >>> from mindquantum.core import Circuit
            >>> circ = Circuit().h(0).x(2, 0).y(3).x(3, 2)
            >>> circ
            q0: ──H────●───────
                       │
            q2: ───────X────●──
                            │
            q3: ──Y─────────X──
            >>> circ.reverse_qubits()
            q0: ──Y─────────X──
                            │
            q1: ───────X────●──
                       │
            q3: ──H────●───────
           