Class mindquantum.io.HiQASM

    将电路转换为hiqasm格式。

    样例:
        >>> import numpy as np
        >>> from mindquantum.io.qasm import HiQASM
        >>> from mindquantum.core import Circuit
        >>> circuit = Circuit().rx(0.3, 0).z(0, 1).zz(np.pi, [0, 1])
        >>> hiqasm = HiQASM()
        >>> circuit_str = hiqasm.to_string(circuit)
        >>> print(circuit_str[68: 80])
        CZ q[1],q[0]
        >>> circuit_2 = hiqasm.from_string(circuit_str)
        >>> circuit_2
        q0: ──RX(3/10)────Z────ZZ(π)──
                          │      │
        q1: ──────────────●────ZZ(π)──
       