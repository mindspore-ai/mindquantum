mindquantum.core.circuit.SwapParts
===================================

.. py:class:: mindquantum.core.circuit.SwapParts(a: Iterable, b: Iterable, maps_ctrl=None)

    交换量子线路中两个不同的部分，可以增加控制比特，也可以不加。

    参数：
        - **a** (Iterable) - 您需要交换的第一部分。
        - **b** (Iterable) - 您需要交换的第二部分。
        - **maps_ctrl** (int, Iterable) - 通过单个量子比特或不同量子比特来控制交换，或者不加控制量子比特。默认值： ``None``。
