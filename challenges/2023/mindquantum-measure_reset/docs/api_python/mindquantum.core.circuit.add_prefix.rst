mindquantum.core.circuit.add_prefix
====================================

.. py:function:: mindquantum.core.circuit.add_prefix(circuit_fn, prefix: str)

    在含参量子线路或含参量子算子（可以生成含参量子线路的函数）的参数上添加前缀。

    参数：
        - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。
        - **prefix** (str) - 添加到每个参数中的前缀。

    返回：
        Circuit，或可以生成Circuit的函数。

    异常：
        - **TypeError** - 如果前缀不是字符串。
        - **TypeError** - `circuit_fn` 不是Circuit或者未返回Circuit。
