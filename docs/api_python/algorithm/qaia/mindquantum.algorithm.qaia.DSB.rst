mindquantum.algorithm.qaia.DSB
===============================

.. py:class:: mindquantum.algorithm.qaia.DSB(J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None)

    离散模拟分叉算法。

    参考文献：`High-performance combinatorial optimization based on classical mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_。

    参数：
        - **J** (Union[numpy.array, csr_matrix]) - 耦合矩阵，维度为 :math:`(N x N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N x batch_size)`。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``0.1``。
        - **xi** (float) - 频率维数，正的常数。默认值： ``None``。

    .. py:method:: update()

        基于修改的显式辛欧拉方法的动力学演化。
