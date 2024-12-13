mindquantum.algorithm.library.qjpeg
===================================

.. py:function:: mindquantum.algorithm.library.qjpeg(n_qubits: int, m_qubits: int)

    基于QJEPG算法实现对量子图像的压缩。

    .. note::
        参数 n_qubits 和 m_qubits 都需要为偶数，且 n_qubits 不小于 m_qubits。更多信息请参考 arXiv:2306.09323v2。

    参数：
        - **n_qubits** (int) - 用于编码待压缩量子图像的量子比特数。
        - **m_qubits** (int) - 用于编码压缩后量子图像的量子比特数。

    返回：
        - Circuit， QJPEG 算法的量子线路
        - List[int]， 保留比特的索引列表，这些比特携带压缩后的量子图像信息
        - List[int]， 丢弃比特的索引列表，这些比特包含原量子图像中的冗余信息
