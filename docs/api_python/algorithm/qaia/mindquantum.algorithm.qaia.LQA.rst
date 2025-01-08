mindquantum.algorithm.qaia.LQA
==============================

.. py:class:: mindquantum.algorithm.qaia.LQA(J, h=None, x=None, n_iter=1000, batch_size=1, gamma=0.1, dt=1.0, momentum=0.99)

    局域量子退火算法。

    参考文献：`Quadratic Unconstrained Binary Optimization via Quantum-Inspired Annealing <https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.18.034016>`_。

    .. note::
        为了内存效率，输入数组 'x' 不会被复制，并且会在优化过程中被原地修改。
        如果需要保留原始数据，请使用 `x.copy()` 传入副本。

    参数：
        - **J** (Union[numpy.array, scipy.sparse.spmatrix]) - 耦合矩阵，维度为 :math:`(N \times N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N \times batch\_size)`。会在优化过程中被修改。如果不提供（``None``），将被初始化为在 [-0.1, 0.1] 范围内均匀分布的随机值。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``1``。
        - **gamma** (float) - 耦合强度。默认值： ``0.1``。
        - **momentum** (float) - 动量系数。默认值： ``0.99``.

    .. py:method:: initialize()

        初始化自旋。

    .. py:method:: update(beta1=0.9, beta2=0.999, epsilon=10e-8)

        Adam动力学演化。

        参数：
            - **beta1** (float) - Beta1参数。默认值： ``0.9``。
            - **beta2** (float) - Beta2参数。默认值： ``0.999``。
            - **epsilon** (float) - Epsilon参数。默认值： ``10e-8``。
