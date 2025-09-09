mindquantum.algorithm.nisq.Ansatz19
===================================

.. py:class:: mindquantum.algorithm.nisq.Ansatz19(n_qubits, depth, prefix='', suffix='')

    Arxiv 论文中所提及的量子线路19。

    .. image:: ./ansatz_images/ansatz19.png
        :height: 180px

    请参考论文 `Expressibility and entangling capability of parameterized quantum circuits for hybrid
    quantum-classical algorithms <https://arxiv.org/abs/1905.10876>`_.

    参数：
        - **n_qubits** (int) - 量子线路的总比特数。
        - **depth** (int) - ansatz 的循环层数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
