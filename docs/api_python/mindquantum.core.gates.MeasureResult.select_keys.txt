mindquantum.core.gates.MeasureResult.select_keys(*keys)

        从该测量容器中选择某些测量关键点。

        参数:
            keys (tuple[str]): 要选择的键。

        样例:
            >>> from mindquantum import Simulator
            >>> from mindquantum import qft, H
            >>> circ = qft(range(2)).measure('q0_0', 0).measure('q1_0', 1)
            >>> circ.h(0).measure('q0_1', 0)
            >>> circ
            q0: ──H────PS(π/2)─────────@────M(q0_0)────H────M(q0_1)──
                          │            │
            q1: ──────────●───────H────@────M(q1_0)──────────────────
            >>> sim = Simulator('projectq', circ.n_qubits)
            >>> res = sim.sampling(circ, shots=500, seed=42)
            >>> new_res = res.select_keys('q0_1', 'q1_0')
            >>> new_res
            shots: 500
            Keys: q1_0 q0_1│0.00   0.068       0.136       0.204       0.272        0.34
            ───────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                         00│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         01│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
                         10│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                           │
                         11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                           │
            {'00': 127, '01': 107, '10': 136, '11': 130}
           