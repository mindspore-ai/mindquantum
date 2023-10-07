mindquantum.core.gates.NoiseGate
================================

.. py:class:: mindquantum.core.gates.NoiseGate(name, n_qubits, obj_qubits=None, ctrl_qubits=None)

    噪声信道。

    参数：
        - **name** (str) - 此门的名称。
        - **n_qubits** (int) - 这个门有多少个量子比特。
        - **obj_qubits** (int, list[int]) - 具体门作用在哪个量子比特上。
        - **ctrl_qubits** (int, list[int]) - 指定控制量子比特。默认值： `None`。

    .. py:method:: on(obj_qubits, ctrl_qubits=None)

        定义门作用于哪个量子比特和控制量子比特。

        参数：
            - **obj_qubits** (int, list[int]) - 指定门作用在哪个量子比特上。
            - **ctrl_qubits** (int, list[int]) - 噪声信道的控制量子比特应该总是 ``None``。
