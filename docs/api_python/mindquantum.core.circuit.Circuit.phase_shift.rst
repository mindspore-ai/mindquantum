.. py:method:: mindquantum.core.circuit.Circuit.phase_shift(para, obj_qubits, ctrl_qubits=None)

    添加一个Phase Shift门。

    **参数：**
    
    - **para** (Union[dict, ParameterResolver]) - `PhaseShift` 门的参数。
    - **obj_qubits** (Union[int, list[int]]) - `PhaseShift` 门的目标量子比特。
    - **ctrl_qubits** (Union[int, list[int]]) - `PhaseShift` 门的控制量子比特。默认值：None。
