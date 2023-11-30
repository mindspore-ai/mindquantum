mindquantum.algorithm.nisq.RYCascade
====================================

.. py:class:: mindquantum.algorithm.nisq.RYCascade(n_qubits: int, depth: int, prefix: str = '', suffix: str = '')

    以 :class:`~.core.gates.RY` 门作为单比特门，以两层线性分布的 CNOT 门作为纠缠门的硬件友好型线路。

    .. image:: ./ansatz_images/RYCascade.png
        :height: 180px

    请参考论文 `Challenges in the Use of Quantum Computing Hardware-Efficient Ansätze in Electronic Structure Theory <https://pubs.acs.org/doi/10.1021/acs.jpca.2c08430>`_.

    参数：
        - **n_qubits** (int) - 量子线路的总比特数。
        - **depth** (int) - ansatz 的循环层数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
