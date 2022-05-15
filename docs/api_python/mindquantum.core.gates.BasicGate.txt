Class mindquantum.core.gates.BasicGate(name, n_qubits, obj_qubits=None, ctrl_qubits=None)

    BasicGate是所有门的基类。

    参数:
        name (str): 此门的名称。
        n_qubits (int): 这个门有多少个量子位。
        obj_qubits (int, list[int]): 具体门作用在哪个量子位上。
        ctrl_qubits (int, list[int]): 指定控制qbits。默认：None。
    