Class mindquantum.io.OpenQASM

    将电路转换为openqasm格式

    样例:
        >>> import numpy as np
        >>> from mindquantum.io.qasm import OpenQASM
        >>> from mindquantum.core import Circuit
        >>> circuit = Circuit().rx(0.3, 0).z(0, 1).zz(np.pi, [0, 1])
        >>> openqasm = OpenQASM()
        >>> circuit_str = openqasm.to_string(circuit)
        >>> circuit_str[47:60]
        'rx(0.3) q[0];'
    