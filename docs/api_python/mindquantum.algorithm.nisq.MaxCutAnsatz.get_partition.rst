.. py:method:: mindquantum.algorithm.nisq.MaxCutAnsatz.get_partition(max_n, weight)

    获取max-cut问题的切割方案。

    **参数：**

    - **max_n** (int) – 需要多少个切割方案。
    - **weight** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) – max-cut ansatz的参数值。

    **返回：**

    list，切割方案构成的列表。
        