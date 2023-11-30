mindquantum.algorithm.nisq.RYLinear
===================================

.. py:class:: mindquantum.algorithm.nisq.RYLinear(n_qubits: int, depth: int, prefix: str = '', suffix: str = '')

    以 :class:`~.core.gates.RY` 门作为单比特门，以线性分布的 CNOT 门作为纠缠门的硬件友好型线路。

    .. image:: ./ansatz_images/RYLinear.png
        :height: 180px

    请参考论文 `Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets <https://www.nature.com/articles/nature23879>`_.

    参数：
        - **n_qubits** (int) - 量子线路的总比特数。
        - **depth** (int) - ansatz 的循环层数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
