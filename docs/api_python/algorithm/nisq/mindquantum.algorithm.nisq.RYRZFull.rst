mindquantum.algorithm.nisq.RYRZFull
===================================

.. py:class:: mindquantum.algorithm.nisq.RYRZFull(n_qubits: int, depth: int, prefix: str = '', suffix: str = '')

    以 :class:`~.core.gates.RY` 门和 :class:`~.core.gates.RZ` 作为单比特门，以两两比特都有作用的 CNOT 门作为纠缠门的硬件友好型线路。

    .. image:: ./ansatz_images/RYRZFull.png
        :height: 180px

    参数：
        - **n_qubits** (int) - 量子线路的总比特数。
        - **depth** (int) - ansatz 的循环层数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
