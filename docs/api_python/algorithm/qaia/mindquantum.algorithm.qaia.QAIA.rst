mindquantum.algorithm.qaia.QAIA
===============================

.. py:class:: mindquantum.algorithm.qaia.QAIA(J, h=None, x=None, n_iter=1000, batch_size=1)

    量子退火启发式算法基类。

    此类包含所有QAIA算法的基本和共同方法接口。

    .. note::
        为了内存效率，输入数组 'x' 不会被复制，并且会在优化过程中被原地修改。
        如果需要保留原始数据，请使用 `x.copy()` 传入副本。

    参数：
        - **J** (Union[numpy.array, scipy.sparse.spmatrix]) - 耦合矩阵，维度为 :math:`(N \times N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N \times batch\_size)`。会在优化过程中被修改。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。

    .. py:method:: calc_cut(x=None)

        计算切割值。

        参数：
          - **x** (numpy.array) - 自旋配置，维度为 :math:`(N \times batch\_size)`。如果为 ``None``，初始自旋将会被使用。默认值： ``None``。

    .. py:method:: calc_energy(x=None)

        计算能量值。

        参数：
          - **x** (numpy.array) - 自旋配置，维度为 :math:`(N \times batch\_size)`。如果为 ``None``，初始自旋将会被使用。默认值： ``None``。

    .. py:method:: initialize()

        随机化初始化自旋。
