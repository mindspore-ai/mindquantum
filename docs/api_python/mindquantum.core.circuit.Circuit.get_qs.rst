.. py:method:: mindquantum.core.circuit.Circuit.get_qs(backend='projectq', pr=None, ket=False, seed=None)

    获取线路的最终量子态。

    **参数：**

    - **backend** (str) - 使用的后端。默认值：'projectq'。
    - **pr** (Union[numbers.Number, ParameterResolver, dict, numpy.ndarray]) - 线路的参数，线路含参数时提供。默认值：None。
    - **ket** (str) - 是否以ket格式返回量子态。默认值：False。
    - **seed **(int) - 模拟器的随机种子。默认值：None。