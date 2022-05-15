Class mindquantum.core.gates.Measure(name='')

    测量量子量子位的测量门。

    参数:
        name (str): 此测量门的键。在量子电路中，不同测量门的密钥应该是唯一的。默认值：“”。

    样例:
        >>> import numpy as np
        >>> from mindquantum import qft, Circuit
        >>> from mindquantum import Measure
        >>> from mindquantum import Simulator
        >>> circ = qft(range(2))
        >>> circ += Measure('q0').on(0)
        >>> circ += Measure().on(1)
        >>> circ
        q0: ──H────PS(π/2)─────────@────M(q0)──
                      │            │
        q1: ──────────●───────H────@────M(q1)──
        >>> sim = Simulator('projectq', circ.n_qubits)
        >>> sim.apply_circuit(Circuit().h(0).x(1, 0))
        >>> sim
        projectq simulator with 2 qubits (little endian).
        Current quantum state:
        √2/2¦00⟩
        √2/2¦11⟩
        >>> res = sim.sampling(circ, shots=2000, seed=42)
        >>> res
        shots: 2000
        Keys: q1 q0│0.00   0.124       0.248       0.372       0.496       0.621
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 10│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
        {'00': 993, '10': 506, '11': 501}
        >>> sim
        projectq simulator with 2 qubits (little endian).
        Current quantum state:
        √2/2¦00⟩
        √2/2¦11⟩
        >>> sim.apply_circuit(circ[:-2])
        >>> sim
        projectq simulator with 2 qubits (little endian).
        Current quantum state:
        √2/2¦00⟩
        (√2/4-√2/4j)¦10⟩
        (√2/4+√2/4j)¦11⟩
        >>> np.abs(sim.get_qs())**2
        array([0.5 , 0.  , 0.25, 0.25])
       