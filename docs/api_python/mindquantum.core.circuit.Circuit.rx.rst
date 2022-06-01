.. py:method:: mindquantum.core.circuit.Circuit.rx(para, obj_qubits, ctrl_qubits=None)

    在电路中添加 `RX` 门。

    **参数：**

    - **para** (Union[dict, ParameterResolver]) - `RX` 门的参数。
    - **obj_qubits** (Union[int, list[int]]) - `RX` 门的目标量子比特。
    - **ctrl_qubits** (Union[int, list[int]]) - `RX` 门的控制量子比特。默认值：None。
