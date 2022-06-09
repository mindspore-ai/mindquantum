.. py:class:: mindquantum.core.circuit.UN(gate: BasicGate, maps_obj, maps_ctrl=None)

    将量子门映射到不同的目标量子位和控制量子位。

    **参数：**

    - **gate** (BasicGate) - 量子门。
    - **maps_obj** (Union[int, list[int]]) - 目标量子位。
    - **maps_ctrl** (Union[int, list[int]]) - 控制量子位。默认值： `None` 。

    **返回：**

    Circuit，返回一个量子电路。
