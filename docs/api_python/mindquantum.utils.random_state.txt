mindquantum.utils.random_state(shapes, norm_axis=0, comp=True, seed=None)

    生成一些随机量子态。

    参数:
        shapes (tuple): shapes = (m, n) 表示m个量子状态，每个状态由 :math:`\log_2(n)` 量子位形成。
        norm_axis (int): 要应用标准化的轴。默认值：0。
        comp (bool): 如果`True`，量子态的每个振幅将是一个复数。默认值：True。
        seed (int): 随机种子。默认值：None。

    返回:
        numpy.nd数组，一种归一化的随机量子态。

    样例:
        >>> from mindquantum.utils import random_state
        >>> random_state((2, 2), seed=42)
        array([[0.44644744+0.18597239j, 0.66614846+0.10930256j],
               [0.87252821+0.06923499j, 0.41946926+0.60691409j]])
    