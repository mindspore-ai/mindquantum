mindquantum.algorithm.library.amplitude_encoder
================================================

.. py:function:: mindquantum.algorithm.library.amplitude_encoder(x, n_qubits)

    用于振幅编码的量子线路。

    .. note::
        经典数据的长度应该是2的幂，否则将用0填充。
        向量应该归一化。

    参数：
        - **x** (list[float] or numpy.array(list[float])) - 需要编码的数据向量，应该归一化。
        - **n_qubits** (int) - 编码线路的量子比特数。

    返回：
        Circuit，能够完成振幅编码的量子线路。
        ParameterResolver，用于完成振幅编码量子线路的参数。
