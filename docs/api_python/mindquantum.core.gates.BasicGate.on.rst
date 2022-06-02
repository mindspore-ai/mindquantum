.. py:method:: mindquantum.core.gates.BasicGate.on(obj_qubits, ctrl_qubits=None)

    定义门作用于哪个量子比特和控制量子比特。

    .. note::
        在此框架中，首先指定门作用的量子位，即使对于控制门，例如CNOT，第二个参数是控制量子位。

    **参数：**

    - **obj_qubits** (int, list[int]) - 指定门作用在哪个量子位上。
    - **ctrl_qubits** (int, list[int]) - 指定控制量子位。默认：None。

    **返回：**

    返回一个新的门。
