mindquantum.algorithm.mapping.SABRE
===================================

.. py:class:: mindquantum.algorithm.mapping.SABRE(circuit: Circuit, topology: QubitsTopology)

    用于比特映射的 SABRE 算法。

    更多关于 SABRE 算法的细节，请参考论文： `Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices <https://arxiv.org/abs/1809.02573>`_。

    参数：
        - **circuit** (:class:`~.core.circuit.Circuit`) - 需要做比特映射的量子线路。当前仅支持单比特或者两比特量子门，且控制为包含在其中。
        - **topology** (:class:`~.device.QubitsTopology`) - 量子硬件的比特拓扑结构。当前仅支持联通图。

    .. py:method:: solve(iter_num: int, w: float, delta1: float, delta2: float)

        利用 SABRE 算法来求解比特映射问题。

        参数：
            - **iter_num** (int) - 求解比特映射是的迭代次数。
            - **w** (float) - w 参数。更多信息，请参考论文。
            - **delta1** (float) - delta1 参数。更多信息，请参考论文。
            - **delta2** (float) - delta2 参数。更多信息，请参考论文。

        返回：
            Tuple[:class:`~.core.circuit.Circuit`, List[int], List[int]]，一个可以在硬件上执行的量子线路，初始的映射顺序，最后的映射顺序。
