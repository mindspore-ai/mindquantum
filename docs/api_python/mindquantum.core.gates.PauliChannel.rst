.. py:class:: mindquantum.core.gates.PauliChannel(px: float, py: float, pz: float, **kwargs)

    量子信道可以描述量子计算中的非相干噪声。

    泡利信道描述的噪声体现为：在量子比特上随机作用一个额外的泡利门，是X、Y和Z门对应概率分别为Px、Py和Pz，或以概率1-Px-Py-Pz的概率保持不变（作用I门）。。

    泡利信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P_x - P_y - P_z)\rho + P_x X \rho X + P_y Y \rho Y + P_z Z \rho Z

    其中，ρ是密度矩阵形式的量子态；Px、Py和Pz是作用的泡利门为X、Y和Z门的概率。

    **参数：**
      - **px** (int, float) - 作用的泡利门是X门的概率。
      - **py** (int, float) - 作用的泡利门是Y门的概率。
      - **pz** (int, float) - 作用的泡利门是Z门的概率。

    **样例：**
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
       