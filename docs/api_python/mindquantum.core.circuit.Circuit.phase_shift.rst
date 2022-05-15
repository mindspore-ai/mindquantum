mindquantum.core.circuit.Circuit.phase_shift(para, obj_qubits, ctrl_qubits=None)

        添加相移门。

        参数:
            para (Union[dict, ParameterResolver]): `PhaseShift`门的参数。
            obj_qubits (Union[int, list[int]]): `相位移'门的对象量子位。
            ctrl_qubits (Union[int, list[int]]): `相位移'门的控制量子位。默认值：None。