.. py:function:: mindquantum.core.circuit.as_ansatz(circuit_fn)

    将线路转化为ansatz线路的装饰器。

    **参数：**

    - **circuit_fn** (Union[Circuit, FunctionType, MethodType]) - 一个线路或着返回值为线路的函数。

    **返回：**

    Function， 如果 ``circuit_fn`` 是一个返回值为线路的函数。Circuit，如果 ``circuit_fn`` 本身就是线路。
