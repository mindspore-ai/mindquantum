src.maxcut
=============

求解最大切割问题。

|

.. py:function:: src.maxcut.func(w, loss, grad=False, show_iter_val=False)

    `spicy.optimize.minimize` 所需的损失函数。

    参数：
        - **w** (Union[numpy.ndarray, list]) - 待优化参数。
        - **loss** (src.mbe_loss.MBELoss) - MBE-VQO的损失函数。
        - **grad** (bool) - 是否返回梯度，默认为False。
        - **show_iter_val** (bool) - 是否打印迭代过程，默认为False。

    返回：
        float，得分。

|

.. py:function:: src.maxcut.maxcut(n, depth, problem, weight=None, grad=False, show_iter_val=False, **kwargs)

    使用 `spicy` 的最小值优化器对MBE-VQO下的MaxCut问题进行求解。

    参数：
        - **n** (int) - 图节点数。
        - **depth** (int) - 量子线路深度。
        - **problem** (Union[numpy.ndarray, list]) - MaxCut问题对应带权图，[[node1, node2, weight], ...]。
        - **weight** (Union[numpy.ndarray, list]) - 初始待优化参数，默认None为随机初始参数。
        - **grad** (bool) - 是否使用梯度，默认为False。
        - **show_iter_val** (bool) - 是否打印迭代过程，默认为False。
        - **kwargs** (dict) - 传入 `spicy.optimize.minimize` 的其他关键字参数。

    返回：
        src.mbe_loss.MBELoss，MBE-VQO的损失函数。numpy.ndarray，优化后参数。numpy.ndarray，问题求解结果。
