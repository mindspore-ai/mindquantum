mindquantum.algorithm.qaia.SimCIM
=================================

.. py:class:: mindquantum.algorithm.qaia.SimCIM(J, h=None, x=None, n_iter=1000, batch_size=1, dt=0.01, momentum=0.9, sigma=0.03, pt=6.5)

    模拟相干伊辛机算法。

    参考文献：`Annealing by simulating the coherent Ising machine <https://opg.optica.org/oe/fulltext.cfm?uri=oe-27-7-10288&id=408024>`_。

    参数：
        - **J** (Union[numpy.array, csr_matrix]) - 耦合矩阵，维度为 :math:`(N x N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N x batch_size)`。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值为： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``1``。
        - **momentum** (float) - 动量系数。默认值： ``0.9``。
        - **sigma** (float) - 噪声标准差。默认值： ``0.03``。
        - **pt** (float) - 泵浦系数。默认值： ``6.5``。

    .. py:method:: initialize()

        初始化自旋。

    .. py:method:: update()

        Adam动力学演化。
