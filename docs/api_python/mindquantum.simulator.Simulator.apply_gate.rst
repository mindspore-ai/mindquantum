.. py:method:: mindquantum.simulator.Simulator.apply_gate(gate, pr=None, diff=False)

    在此模拟器上应用门，可以是量子门或测量算子。

    **参数：**

    - **gate** (BasicGate) - 要应用的门。
    - *pr** (Union[numbers.Number, numpy.ndarray, ParameterResolver, list]) – 含参门的参数。默认值：None。
    - **diff** (bool) – 是否在模拟器上应用导数门。默认值：False。

    **返回：**

    int或None，如果是该门是测量门，则返回坍缩态，否则返回None。

    **异常：**

    - **TypeError** – 如果 `gate` 不是BasicGate。
    - **ValueError** – 如果 `gate` 的任何量子位高于模拟器量子位。
    - **ValueError** – 如果 `gate` 是含参的，但没有提供参数。
    - **TypeError** – 如果 `gate` 是含参的，但 `pr` 不是ParameterResolver。                