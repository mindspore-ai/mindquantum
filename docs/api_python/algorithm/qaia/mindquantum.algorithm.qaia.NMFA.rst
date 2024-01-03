mindquantum.algorithm.qaia.NMFA
===============================

.. py:class:: mindquantum.algorithm.qaia.NMFA(J, h=None, x=None, n_iter=1000, batch_size=1, alpha=0.15, sigma=0.15)

    含噪平均场退火算法。

    参考文献：`Emulating the coherent Ising machine with a mean-field algorithm <https://arxiv.org/abs/1806.08422>`_。

    参数：
        - **J** (Union[numpy.array, csr_matrix]) - 耦合矩阵，维度为 :math:`(N x N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N x batch_size)`。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **alpha** (float) - 动量系数。默认值： ``0.15``。
        - **sigma** (float) - 噪声标准差。默认值： ``0.15``。

    .. py:method:: initialize()

        初始化自旋。

    .. py:method:: update()

        动力学演化。
