.. py:method:: mindquantum.core.circuit.Circuit.measure(key, obj_qubit=None)

    添加一个测量门。

    **参数：**

    - **key** (Union[int, str]) - 如果 `obj_qubit` 为None，则 `key` 应为int，表示要测量哪个量子比特，否则， `key` 应为str，表示测量门的名称。
    - **obj_qubit** (int) - 要测量的量子比特。默认值：None。
