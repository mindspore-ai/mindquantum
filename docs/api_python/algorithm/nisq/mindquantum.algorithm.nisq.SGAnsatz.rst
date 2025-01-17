mindquantum.algorithm.nisq.SGAnsatz
========================================

.. py:class:: mindquantum.algorithm.nisq.SGAnsatz(nqubits, k, nlayers=1, prefix: str = '', suffix: str = '')

    序列生成（SG）ansatz，用于一维量子系统。

    SG ansatz由多个变分量子电路块组成，每个块都是应用于相邻量子比特的参数化量子电路。这种结构使得SG ansatz天然适合于量子多体问题。

    具体而言，对于一维量子系统，SG ansatz可以高效地生成具有固定键维度的任意矩阵乘积态。对于二维系统，SG ansatz可以生成 string-bond 态。

    了解更多详细信息，请访问 `A sequentially generated variational quantum circuit with polynomial complexity <https://arxiv.org/abs/2305.12856>`_。

    参数：
        - **nqubits** (int) - ansatz中的量子比特数。
        - **k** (int) - log(R) + 1，其中R是MPS态的键维度。
        - **nlayers** (int) - 每个块中的层数。默认值：``1``。
        - **prefix** (str) - 参数的前缀。默认值：``''``。
        - **suffix** (str) - 参数的后缀。默认值：``''``。
