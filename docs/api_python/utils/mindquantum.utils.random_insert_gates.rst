mindquantum.utils.random_insert_gates
======================================

.. py:function:: mindquantum.utils.random_insert_gates(circuit: Circuit, gates: BasicGate | list[BasicGate], nums: int | list[int], focus_on: int | list[int] | None = None, with_ctrl: bool = True, after_measure: bool = False, shots: int = 1, seed: int | None = None)

    将指定数量的单量子比特门插入到量子线路中的随机位置。

    参数：
        - **circuit** (Circuit) - 待随机插入量子门的量子线路。
        - **gates** (Union[BasicGate, List[BasicGate]]) - 所有待插入的量子门。
        - **nums** (Union[int, List[int]]) - 每种量子门插入的数量。
        - **focus_on** (Union[int, List[int]], 可选) - 只将量子门作用在 focus_on 比特上。如果为 None，则作用在量子线路的所有比特上。默认值： None。
        - **with_ctrl** (bool, 可选) - 是否在控制位上插入量子门。默认值： True。
        - **after_measure** (bool, 可选) - 是否在测量门后插入量子门。默认值： False。
        - **shots** (int, 可选) - 随机采样量子线路的数量。默认值： 1。
        - **seed** (int, 可选) - 生成插入位置的随机数种子。默认值： None。

    返回：
        一个可以产生量子线路的生成器。