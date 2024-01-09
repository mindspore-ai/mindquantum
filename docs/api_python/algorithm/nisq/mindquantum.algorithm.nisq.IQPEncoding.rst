mindquantum.algorithm.nisq.IQPEncoding
=======================================

.. py:class:: mindquantum.algorithm.nisq.IQPEncoding(n_feature, first_rotation_gate=RZ, second_rotation_gate=RZ, num_repeats=1, prefix: str = '', suffix: str = '')

    通用IQP编码。

    更多信息请参考 `Supervised learning with quantum-enhanced feature spaces. <https://www.nature.com/articles/s41586-019-0980-2>`_。

    参数：
        - **n_feature** (int) - IQP编码所需编码的数据的特征数。
        - **first_rotation_gate** (ParameterGate) - 旋转门RX、RY或RZ之一。
        - **second_rotation_gate** (ParameterGate) - 旋转门RX、RY或RZ之一。
        - **num_repeats** (int) - 编码迭代次数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。

    .. py:method:: data_preparation(data)

        IQP编码的ansatz能够将经典数据编码为量子态。
        这种方法将经典数据准备成适合IQP编码的维数。
        假设源数据具有 :math:`n` 特征，那么输出数据将具有 :math:`2n-1` 特征，前 :math:`n` 个特征是原始数据。对于 :math:`m > n` 。

        .. math::

            \text{data}_m = \text{data}_{m - n} * \text{data}_{m - n - 1}

        参数：
            - **data** ([list, numpy.ndarray]) - IQP编码了解更多详细信息所需要的经典数据。

        返回：
            numpy.ndarray，适合此ansatz维度的数据。
