.. py:function:: mindquantum.core.circuit.dagger(circuit_fn)

    获取量子线路或量子算子的共轭转置(dagger)（是一个可以生成量子线路的函数）。

    **参数：**

    - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。

    **异常：**

    - **TypeError** - 如果circuit_fn不是Circuit或无法返回Circuit。

    **返回：**

    Circuit或可以生成Circuit的函数。
