mindquantum.core.circuit.dagger
================================

.. py:function:: mindquantum.core.circuit.dagger(circuit_fn)

    获取量子线路或量子算子（可以生成量子线路的函数）的共轭转置。

    参数：
        - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。

    返回：
        Circuit，或可以生成Circuit的函数。

    异常：
        - **TypeError** - 如果 `circuit_fn` 不是Circuit或无法返回Circuit。
