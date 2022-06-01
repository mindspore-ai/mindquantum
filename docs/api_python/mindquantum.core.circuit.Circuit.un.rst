.. py:method:: mindquantum.core.circuit.Circuit.un(gate, maps_obj, maps_ctrl=None)

    将量子门作用于不同的目标量子比特和控制量子比特，详见类 :class:`mindquantum.core.circuit.UN` 。

    **参数：**

    - **gate** (BasicGate) - 要执行的量子门。
    - **map_obj** (Union[int, list[int]]) - 执行该量子门的目标量子比特。
    - **maps_ctrl** (Union[int, list[int]]) - 执行该量子门的控制量子比特。默认值：None。
