Class mindquantum.core.gates.PauliChannel(px: float, py: float, pz: float, **kwargs)

    量子信道，表达量子计算中的非相干噪声。

    Pauli通道表示误差，在具有不同概率Px、Py和Pz的量子比特上随机应用额外的X、Y或Z门，或注意（应用I门），概率P=(1-Px-Py-Pz)。

    Pauli通道将噪声应用为：

    .. math::

        \epsilon(\rho) = (1 - P_x - P_y - P_z)\rho + P_x X \rho X + P_y Y \rho Y + P_z Z \rho Z

    其中，，是密度矩阵类型的量子态；
    Px、Py和Pz是应用额外X、Y和Z栅极的概率。

    参数:
        px (int, float): 应用X门的概率。
        py (int, float): 应用Y门的概率。
        pz (int, float): 应用Z门的概率。

    样例:
        >>> from mindquantum.core.gates import PauliChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PauliChannel(0.8, 0.1, 0.1).on(0)
        >>> circ += PauliChannel(0, 0.05, 0.9).on(1, 0)
        >>> circ.measure_all()
        >>> print(circ)
        q0: ──PC────●─────M(q0)──
                    │
        q1: ────────PC────M(q1)──
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('projectq', 2)
        >>> sim.sampling(circ, shots=1000, seed=42)
        shots: 1000
        Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▒▒▒▒▒▒▒
                   │
                 01│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 11│▒▒▒
                   │
        {'00': 101, '01': 862, '11': 37}
       