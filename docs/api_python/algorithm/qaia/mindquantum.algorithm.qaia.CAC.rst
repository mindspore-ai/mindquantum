mindquantum.algorithm.qaia.CAC
==============================

.. py:class:: mindquantum.algorithm.qaia.CAC(J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.075)

    混沌振幅控制算法。

    参考文献：`Coherent Ising machines with optical error correction circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_。

    参数：
        - **J** (Union[numpy.array, csr_matrix]) - 耦合矩阵，维度为 :math:`(N x N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N x batch_size)`。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``0.075``。

    .. py:method:: initialize()

        初始化自旋和错误变量。

    .. py:method:: update()

        动力学演化。
