mindquantum.core.circuit.as_ansatz
===================================

.. py:function:: mindquantum.core.circuit.as_ansatz(circuit_fn)

    将线路转化为ansatz线路的装饰器。

    参数：
        - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。

    返回：
        - Function， 如果 `circuit_fn` 是一个返回值为 `Circuit` 的函数。
        - Circuit，如果 `circuit_fn` 本身就是 `Circuit` 。
