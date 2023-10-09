mindquantum.core.circuit.BitFlipAdder
=====================================

.. py:class:: mindquantum.core.circuit.BitFlipAdder(flip_rate: float, with_ctrl=True, focus_on: int = None, add_after: bool = True)

    在量子门前面或者后面添加一个比特翻转信道。

    参数：
        - **flip_rate** (float) - 比特翻转信道的翻转概率。具体请参考 :class:`~.core.gates.BitFlipChannel`。
        - **with_ctrl** (bool) - 是否在控制为上添加比特。默认值： ``True``。
        - **focus_on** (int) - 只讲该噪声信道作用在 ``focus_on`` 比特上。如果为 ``None``，则作用在量子门的所有比特上。默认值： ``None``。
        - **add_after** (bool) - 是否在量子门后面添加信道。如果为 ``False``，信道将会加在量子门前面。默认值： ``True``。
