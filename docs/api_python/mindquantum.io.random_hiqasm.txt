mindquantum.io.random_hiqasm(n_qubits, gate_num, version='0.1', seed=42)

    生成随机hiqasm支持的电路。

    参数:
        n_qubits (int): 此量子电路中的量子比特总数。
        gate_num (int): 此量子电路中的门总数。
        version (str): HIQASM的版本。默认值：“0.1”。
        seed (int): 生成此随机量子电路的随机种子。默认值：42。

    返回:
        str，HIQASM格式的量子。

    样例:
        >>> from mindquantum.io.qasm import random_hiqasm
        >>> from mindquantum.io.qasm import HiQASM
        >>> hiqasm_str = random_hiqasm(2, 5)
        >>> hiqasm = HiQASM()
        >>> circuit = hiqasm.from_string(hiqasm_str)
        >>> circuit
        q0: ──RZ(-2.513)────RZ(-3.012)────RX(0.738)────M(k0)───────────
                                              │
        q1: ──────S───────────────────────────●──────────Z──────M(k1)──
       