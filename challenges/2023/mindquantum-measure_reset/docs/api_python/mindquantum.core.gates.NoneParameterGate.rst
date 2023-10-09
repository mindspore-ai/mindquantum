mindquantum.core.gates.NoneParameterGate
=========================================

.. py:class:: mindquantum.core.gates.NoneParameterGate(name, n_qubits, obj_qubits=None, ctrl_qubits=None)

    非参数化的门。

    参数：
        - **name** (str) - 此门的名称。
        - **n_qubits** (int) - 这个门有多少个量子比特。
        - **obj_qubits** (int, list[int]) - 具体门作用在哪个量子比特上。
        - **ctrl_qubits** (int, list[int]) - 指定控制量子比特。默认值： `None`。
