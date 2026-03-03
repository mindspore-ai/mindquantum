mindquantum.algorithm.qaia.LSB
===============================

.. py:class:: mindquantum.algorithm.qaia.LSB(J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None, backend='cpu-float32', threshold=8)

    对数模拟分岔算法。

    参考文献：`Quantized Simulated Bifurcation for the Ising Model <https://ieeexplore.ieee.org/document/10231308>`_。

    .. note::
        为了内存效率，输入数组 'x' 不会被复制，并且会在优化过程中被原地修改。
        如果需要保留原始数据，请使用 `x.copy()` 传入副本。

    参数：
        - **J** (Union[numpy.array, scipy.sparse.spmatrix]) - 耦合矩阵，维度为 :math:`(N \times N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N \times batch\_size)`。会在优化过程中被修改。如果不提供（``None``），将被初始化为在 [-0.01, 0.01] 范围内均匀分布的随机值。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``1``。
        - **xi** (float) - 频率维数，正的常数。默认值： ``None``。
        - **backend** (str) - 计算后端和精度：'cpu-float32'、'gpu-float32'。默认值： ``'cpu-float32'``。
        - **threshold** (int) - 动态阈值。默认值： ``8``。

    .. py:method:: update()

        基于修改的显式辛欧拉方法的动力学演化。
