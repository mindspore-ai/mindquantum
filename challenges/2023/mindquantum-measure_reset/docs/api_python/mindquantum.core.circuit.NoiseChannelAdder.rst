mindquantum.core.circuit.NoiseChannelAdder
==========================================

.. py:class:: mindquantum.core.circuit.NoiseChannelAdder(channel: NoiseGate, with_ctrl=True, focus_on: int = None, add_after: bool = True)

    添加一个单比特量子信道。

    参数：
        - **channel** (:class:`~.core.gates.NoiseGate`) - 一个单比特量子信道。
        - **with_ctrl** (bool) - 是否在控制为上添加比特。默认值： ``True``。
        - **focus_on** (int) - 只讲该噪声信道作用在 ``focus_on`` 比特上。如果为 ``None``，则作用在量子门的所有比特上。默认值： ``None``。
        - **add_after** (bool) - 是否在量子门后面添加信道。如果为 ``False``，信道将会加在量子门前面。默认值： ``True``。
