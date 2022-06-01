.. py:method:: mindquantum.core.circuit.Circuit.rz(para, obj_qubits, ctrl_qubits=None)

    在电路中添加 `RZ` 门。

    **参数：**

    - **para** (Union[dict, ParameterResolver]) - `RZ` 门的参数。
    - **obj_qubits** (Union[int, list[int]]) - `RZ` 门的目标量子比特。
    - **ctrl_qubits** (Union[int, list[int]]) - `RZ` 门的控制量子比特。默认值：None。
