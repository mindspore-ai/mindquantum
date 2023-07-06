mindquantum.core.circuit.BitFlipAdder
========================================

.. py:class:: mindquantum.core.circuit.BitFlipAdder(flip_rate: float, with_ctrl=True, add_after: bool = True)

    在量子门前面或者后面添加一个比特翻转信道。

    参数：
        - **flip_rate** (float) - 比特翻转信道的翻转概率。具体请参考 :class:`~.core.gates.BitFlipChannel`。
        - **with_ctrl** (bool) - 是否在控制为上添加比特。默认值： ``True``。
        - **add_after** (bool) - 是否在量子门后面添加信道。如果为 ``False``，信道将会加在量子门前面。默认值： ``True``。
