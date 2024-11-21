mindquantum.algorithm.library.qjpeg
===================================

.. py:function:: mindquantum.algorithm.library.qjpeg(n_qubits, m_qubits)

    基于QJEPG算法实现对量子图像的压缩。

    .. note::
        参数 n_qubits 和 m_qubits 都需要为偶数，且 n_qubits 不小于 m_qubits。更多信息请参考 arXiv:2306.09323v2。

    参数：
        - **n_qubits** (int) - 用于编码待压缩量子图像的量子比特数。
        - **m_qubits** (int) - 用于编码压缩后量子图像的量子比特数。

    返回：
        - Circuit，QJPEG 算法的量子线路。
        - list[int]，携带着压缩后量子图像信息的量子比特。
        - list[int]，被抛弃的量子比特，这些量子比特中含有原量子图像中的冗余信息。
