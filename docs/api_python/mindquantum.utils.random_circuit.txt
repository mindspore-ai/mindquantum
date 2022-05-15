mindquantum.utils.random_circuit(n_qubits, gate_num, sd_rate=0.5, ctrl_rate=0.2, seed=None)

    生成随机电路。

    参数:
        n_qubits (int): 随机电路的量子位数。
        gate_num (int): 随机电路中门的数量。
        sd_rate (float): 单量子位门和双量子位门的速率。
        ctrl_rate (float): 门具有控制量子位的可能性。
        seed (int): 生成随机电路的随机种子。

    样例:
        >>> from mindquantum.utils import random_circuit
        >>> random_circuit(3, 4, 0.5, 0.5, 100)
        q1: ──Z────RX(0.944)────────●────────RX(-0.858)──
              │        │            │            │
        q2: ──●────────●────────RZ(-2.42)────────●───────
       