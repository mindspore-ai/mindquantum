.. py:method:: mindquantum.simulator.Simulator.apply_circuit(circuit, pr=None)

    在模拟器上应用量子线路。

    **参数：**

    - **circuit** (Circuit) - 要应用在模拟器上的量子线路。
    - **pr** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) - 线路的ParameterResolver。如果线路不含参数，则此参数应为None。默认值：None。

    **返回：**

    MeasureResult或None，如果线路具有测量门，则返回MeasureResult，否则返回None。           
