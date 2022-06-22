.. py:function:: mindquantum.core.circuit.apply(circuit_fn, qubits)

    将量子线路或量子算子（可以生成量子线路的函数）作用到不同的量子比特上。

    **参数：**

    - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 量子线路，或可以生成量子线路的函数。
    - **qubits** (list[int]) - 要应用的新量子位。

    **异常：**

    - **TypeError** - 如果量子比特不是list。
    - **ValueError** - 如果量子比特的任何元素为负数。
    - **TypeError** - 如果circuit_fn不是Circuit或不返回Circuit。

    **返回：**

    Circuit或可以生成Circuit的函数。
