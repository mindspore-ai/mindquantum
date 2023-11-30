mindquantum.algorithm.nisq.ASWAP
================================

.. py:class:: mindquantum.algorithm.nisq.ASWAP(n_qubits: int, depth: int, prefix: str = '', suffix: str = '')

    类 SWAP 门形式的硬件友好型线路

    .. image:: ./ansatz_images/ASWAP.png
        :height: 180px

    请参考论文 `Efficient symmetry-preserving state preparation circuits for the variational quantum eigensolver algorithm <https://www.nature.com/articles/s41534-019-0240-1>`_.

    参数：
        - **n_qubits** (int) - 量子线路的总比特数。
        - **depth** (int) - ansatz 的循环层数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
