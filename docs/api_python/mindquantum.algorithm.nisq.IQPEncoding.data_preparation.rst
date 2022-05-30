.. py:method:: mindquantum.algorithm.nisq.IQPEncoding.data_preparation(data)

    IQPEncoding的ansatz能够将经典数据编码为量子态。
    这种方法将经典数据准备成适合IQPEncoding的维数。
    假设源数据具有 :math:`n` 特征，那么输出数据将具有 :math:`2n-1` 特征，第一个 :math:`n` 特征保持不变，对于 :math:`m>n` 。

    .. math::

    \text{data}_m = \text{data}_{m - n} * \text{data}_{m - n - 1}

    **参数：**

    - **data** ([list, numpy.ndarray]) – IQPEncoding所需要的经典数据。

    **返回：**

    numpy.ndarray，适合此ansatz维度的数据。
        