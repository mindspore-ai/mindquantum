mindquantum.algorithm.qaia.CAC
==============================

.. py:class:: mindquantum.algorithm.qaia.CAC(J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.075)

    混沌振幅控制算法。

    参考文献：`Coherent Ising machines with optical error correction circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_。

    .. note::
        为了内存效率，输入数组 'x' 不会被复制，并且会在优化过程中被原地修改。
        如果需要保留原始数据，请使用 `x.copy()` 传入副本。

    参数：
        - **J** (Union[numpy.array, scipy.sparse.spmatrix]) - 耦合矩阵，维度为 :math:`(N \times N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N \times batch\_size)`。会在优化过程中被修改。如果不提供（``None``），将被初始化为服从正态分布 N(0, 10^(-4)) 的随机值。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``0.075``。

    .. py:method:: initialize()

        初始化自旋和错误变量。

    .. py:method:: update()

        动力学演化。
