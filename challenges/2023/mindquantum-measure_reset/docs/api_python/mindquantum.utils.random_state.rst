mindquantum.utils.random_state
===============================

.. py:function:: mindquantum.utils.random_state(shapes, norm_axis=0, comp=True, seed=None)

    生成某个随机的量子态。

    参数：
        - **shapes** (tuple) - 想要生成量子态的个数和维度，例如， `(m, n)` 表示m个量子态，每个状态由 :math:`\log_2(n)` 量子比特形成。
        - **norm_axis** (int) - 应用归一化的轴。默认值： ``0``。
        - **comp** (bool) - 如果为 ``True`` ，量子态的每个振幅将是一个复数。默认值： ``True``。
        - **seed** (int) - 随机种子。默认值： ``None``。

    返回：
        numpy.ndarray，一个随机的归一化量子态。
