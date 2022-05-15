mindquantum.core.circuit.Circuit.un(gate, maps_obj, maps_ctrl=None)

        将量子门映射到不同的目标量子位和控制量子位。请参考UN.

        参数:
            gate (BasicGate): 要映射的基本门。
            map_obj (Union[int, list[int]]): 对象量子位。
            maps_ctrl (Union[int, list[int]]): 控制量子位。默认值：None。
        