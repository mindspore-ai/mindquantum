mindquantum.core.gates.BasicGate.on(obj_qubits, ctrl_qubits=None)

        定义栅极作用于哪个量子位和控制量子位。

        注:
            在此框架中，门作用的量子位首先被指定，即使对于控制门，例如CNOT，第二个参数是控制量子位。

        参数:
            obj_qubits (int, list[int]): 特定的量子位，门作用在哪个量子位上。
            ctrl_qubits (int, list[int]): 指定控制qbits。默认：None。

        返回:
            大门，返回一个新的大门。

        样例:
            >>> from mindquantum.core.gates import X
            >>> x = X.on(1)
            >>> x.obj_qubits
            [1]
            >>> x.ctrl_qubits
            []

            >>> x = X.on(2, [0, 1])
            >>> x.ctrl_qubits
            [0, 1]
        