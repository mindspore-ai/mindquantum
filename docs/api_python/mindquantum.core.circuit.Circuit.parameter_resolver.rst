.. py:method:: mindquantum.core.circuit.Circuit.parameter_resolver()

    获取整个线路的parameter resolver。

    .. note::
        因为相同的参数可以在不同的门中，并且系数可以不同，所以这个parameter resolver只返回量子线路的参数是什么，哪些参数需要梯度。显示系数的更详细的parameter resolver位于线路的每个门中。

    **返回：**

    ParameterResolver，整个线路的parameter resolver。
