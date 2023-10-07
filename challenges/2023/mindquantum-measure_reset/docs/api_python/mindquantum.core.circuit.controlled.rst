mindquantum.core.circuit.controlled
====================================

.. py:function:: mindquantum.core.circuit.controlled(circuit_fn)

    在量子线路或量子算子（可以生成量子线路的函数）上添加控制量子比特。

    参数：
        - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。

    返回：
        可以生成Circuit的函数。

    异常：
        - **TypeError** - `circuit_fn` 不是Circuit或无法返回Circuit。
