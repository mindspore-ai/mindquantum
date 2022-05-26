.. py:method:: mindquantum.algorithm.nisq.MaxCutAnsatz.get_partition(max_n, weight)

    获取max-cut问题的分区。

    **参数：**

    - **max_n** (int) – 需要多少分区。
    - **weight** (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]) – max-cut ansatz的参数值。

    **返回：**

    list，分区列表。
        