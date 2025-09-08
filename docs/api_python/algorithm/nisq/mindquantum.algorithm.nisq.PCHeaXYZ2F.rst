mindquantum.algorithm.nisq.PCHeaXYZ2F
=====================================

.. py:class:: mindquantum.algorithm.nisq.PCHeaXYZ2F(n_qubits, depth, prefix='', suffix='')

    PCHeaXYZ2F 硬件友好型线路。

    .. image:: ./ansatz_images/PCHeaXYZ2F.png
        :height: 180px

    请参考论文 `Physics-Constrained Hardware-Efficient Ansatz on Quantum Computers that is Universal, Systematically Improvable, and Size-consistent <https://arxiv.org/abs/2307.03563>`_.

    参数：
        - **n_qubits** (int) - 量子线路的总比特数。
        - **depth** (int) - ansatz 的循环层数。
        - **prefix** (str) - 参数的前缀。默认值： ``''``。
        - **suffix** (str) - 参数的后缀。默认值： ``''``。
