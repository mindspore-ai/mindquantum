mindquantum.core.circuit.Circuit.remove_measure_on_qubits(qubits)

        删除某些量子位上的所有测量门。

        参数:
            qubit (Union[int, list[int]]): 要删除度量的量子位。

        样例:
            >>> from mindquantum import UN, H, Measure
            >>> circ = UN(H, 3).x(0, 1).x(1, 2).measure_all()
            >>> circ += H.on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ.remove_measure_on_qubits(0)
            q0: ──H────X────H───────────
                       │
            q1: ──H────●────X────M(q1)──
                            │
            q2: ──H─────────●────M(q2)──
           