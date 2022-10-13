src.mbe\_loss
================

损失函数。

|

.. py:class:: src.mbe_loss.MBEEdge(nodes, weight)

    记录无向带权图的边。

    参数：
        - **nodes** (Union[numpy.ndarray, list]) - 节点。
        - **weight** (float) - 权重。

    .. py:method:: get_loss(et, grad)

        计算该边的损失值与梯度。

        参数：
            - **et** (Union[numpy.ndarray, list]) - 忘了。
            - **grad** (bool) - 是否计算梯度。

        返回：
            float，损失值。Union[numpy.ndarray, list, float, int]，梯度。

|

.. py:class:: src.mbe_loss.MBEGraph(n, g)

    记录无向带权图。

    参数：
        - **n** (int) - 图节点数。
        - **g** (Union[numpy.ndarray, list]) - 图。

    .. py:method:: get_loss(et, grad)

        计算该图的损失值与梯度。

        参数：
            - **et** (Union[numpy.ndarray, list]) - 也忘了。
            - **grad** (bool) - 是否计算梯度。

        返回：
            float，损失值。Union[numpy.ndarray, list, float, int]，梯度。

    .. py:method:: build_graph(g)

        内建构图子方法。

        参数：
            - **g** (Union[numpy.ndarray, list]) - 图。

|

.. py:class:: src.mbe_loss.MBELoss(n, depth)

    MBE(Multi-Basis Encodings)方法求解MaxCut问题时的损失值计算器。

    .. math::

            \mathcal{L}_{MBE} = \displaystyle\sum_{j<i}^{n/2} w_{ij}^{zz} tanh(\left<\sigma_i^z\right>) tanh(\left<\sigma_j^z\right>)

    .. math::

			+\displaystyle\sum_{j<i}^{n/2} w_{ij}^{xx} tanh(\left<\sigma_i^x\right>) tanh(\left<\sigma_j^x\right>)

    .. math::

			+\displaystyle\sum_{i,j}^{n/2} w_{ij}^{zx} tanh(\left<\sigma_i^z\right>) tanh(\left<\sigma_j^x\right>)

    参数：
        - **n** (int) - 图节点数。
        - **depth** (int) - 量子线路深度。

    .. py:method:: set_graph(g)

        绑定待求解问题对应图。

        参数：
            - **g** (Union[numpy.ndarray, list]) - 图。

    .. py:method:: get_loss(w, grad=False)

        计算该问题当前的损失值与梯度。

        参数：
            - **w** (Union[numpy.ndarray, list]) - 量子线路参数。
            - **grad** (bool) - 是否计算梯度，默认为False。

        返回：
            float，损失值。Union[numpy.ndarray, list, float, int]，梯度。

    .. py:method:: measure(w)

        按照MBE方法测量量子线路在泡利Z和X方向上的结果。

        参数：
            - **w** (Union[numpy.ndarray, list]) - 量子线路参数。

        返回：
            tuple[numpy.ndarray]，测量结果，第一项为Z测量结果，第二项为X测量结果。
