mindquantum.core.circuit.add_suffix
====================================

.. py:function:: mindquantum.core.circuit.add_suffix(circuit_fn, suffix: str)

    在含参量子线路或含参量子算子（可以生成含参量子线路的函数）的参数上添加后缀。

    参数：
        - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。
        - **suffix** (str) - 添加到每个参数中的后缀。

    返回：
        Circuit，或可以生成Circuit的函数。

    异常：
        - **TypeError** - 如果后缀不是字符串。
        - **TypeError** - `circuit_fn` 不是Circuit或者未返回Circuit。
