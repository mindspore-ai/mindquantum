mindquantum.algorithm.error_mitigation.virtual_distillation
============================================================

.. py:function:: mindquantum.algorithm.error_mitigation.virtual_distillation(circ: Circuit, executor: Callable[[Circuit], Dict[str, int]], little_endian: bool = True, **kwargs)

    基于虚拟蒸馏的误差缓解算法（arXiv:2011.07064）。

    该算法用于计算每个量子比特i上 :math:`Z_i` 泡利算符的误差缓解期望值。要测量其他泡利算符( :math:`X_i` 或 :math:`Y_i`` )的期望值，需要在输入电路末尾添加适当的基矢旋转门：

    - 对于 :math:`X_i` 测量：在量子比特i上添加H门
    - 对于 :math:`Y_i` 测量：在量子比特i上添加RX(π/2)门

    参数：
        - **circ** (:class:`~.core.circuit.Circuit`) - 待执行的量子线路。
        - **executor** (Callable[[:class:`~.core.circuit.Circuit`], dict[str, int]]) - 一个可调用对象，用于执行量子线路并返回一个字典，该字典将测量结果比特串映射到其计数。注意：executor必须能够处理输入线路两倍数量的量子比特。
        - **little_endian** (bool) - executor返回的比特串是否为小端序。默认值：``True``。
        - **kwargs** - 传递给executor的额外参数。

    返回：
        numpy.ndarray，每个量子比特i的误差缓解期望值 :math:`\langle Z_i\rangle` 。要获取其他泡利算符的期望值，需要在调用此函数之前向输入线路添加适当的基矢旋转门。
