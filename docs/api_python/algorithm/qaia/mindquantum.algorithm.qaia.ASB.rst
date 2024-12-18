mindquantum.algorithm.qaia.ASB
===============================

.. py:class:: mindquantum.algorithm.qaia.ASB(J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None, M=2)

    绝热模拟分叉算法。

    参考文献：`Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems <https://www.science.org/doi/10.1126/sciadv.aav2372>`_。

    .. note::
        为了内存效率，输入数组 'x' 不会被复制，并且会在优化过程中被原地修改。
        如果需要保留原始数据，请使用 `x.copy()` 传入副本。

    参数：
        - **J** (Union[numpy.array, scipy.sparse.spmatrix]) - 耦合矩阵，维度为 :math:`(N \times N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N \times batch\_size)`。会在优化过程中被修改。如果不提供（``None``），将被初始化为在 [-0.01, 0.01] 范围内均匀分布的随机值。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``0.1``。
        - **xi** (float) - 频率维数，正的常数。默认值： ``None``。
        - **M** (int) - 不含平均场项的迭代次数。默认值： ``2``。

    .. py:method:: update()

        基于修改的显式辛欧拉方法的动力学演化。
