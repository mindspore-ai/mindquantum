mindquantum.algorithm.qaia.BSB
===============================

.. py:class:: mindquantum.algorithm.qaia.BSB(J, h=None, x=None, n_iter=1000, batch_size=1, dt=1, xi=None, backend='cpu-float32', *, device_id=0)

    弹道模拟分叉算法。

    参考文献：`High-performance combinatorial optimization based on classical mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_。

    .. note::
        为了内存效率，输入数组 'x' 不会被复制，并且会在优化过程中被原地修改。
        如果需要保留原始数据，请使用 `x.copy()` 传入副本。

        当使用backend='gpu-int8'时，请注意该后端可能在稠密图或具有连续系数的图上表现不佳。
        在这些情况下，可以尝试调整参数或考虑使用'cpu-float32'或'gpu-float16'后端。

        Ascend CANN 后端不修改传入的初态数组，而是将 FP32 结果赋给 ``self.x`` 和 ``self.y``。
        首次 update 捕获执行图，后续调用复用；J/h/p 内容或积分参数变更时重建。
        设备资源由算法对象持有，在对象销毁时自动释放。
        ``initialize()`` 不丢弃计划，仅在 ``x is None`` 时重新生成 x，总会重新生成 y。
        J 必须对称且对角为零，默认 xi 包含外场修正；状态始终为二维数组。
        不支持同一对象上的并发修改或更新；并发求解请使用独立实例。

    参数：
        - **J** (Union[numpy.array, scipy.sparse.spmatrix]) - 耦合矩阵，维度为 :math:`(N \times N)`。
        - **h** (numpy.array) - 外场强度，维度为 :math:`(N, )`。
        - **x** (numpy.array) - 自旋初始化配置，维度为 :math:`(N \times batch\_size)`。会在优化过程中被修改。如果不提供（``None``），将被初始化为在 [-0.01, 0.01] 范围内均匀分布的随机值。默认值： ``None``。
        - **n_iter** (int) - 迭代步数。默认值： ``1000``。
        - **batch_size** (int) - 样本个数。默认值： ``1``。
        - **dt** (float) - 迭代步长。默认值： ``1``。
        - **xi** (float) - 频率维数，正的常数。默认值： ``None``。
        - **backend** (str) - 计算后端和精度：'cpu-float32'、'gpu-float32'、'gpu-float16'、'gpu-int8'、'npu-float32' 或 'ascend-float32'。默认值： ``'cpu-float32'``。'ascend-float32' 直接使用 CANN，以 FP32 执行，无需 PyTorch。
        - **device_id** (int) - Ascend CANN 的 ACL 逻辑设备号，仅限关键字参数。其他后端只接受默认值 ``0``。

    .. py:method:: update()

        基于修改的显式辛欧拉方法的动力学演化。
