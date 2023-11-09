mindquantum.algorithm.mapping.MQSABRE
=====================================

.. py:class:: mindquantum.algorithm.mapping.MQSABRE(circuit: Circuit, topology: QubitsTopology, cnoterrorandlength: List[Tuple[Tuple[int, int], List[float]]])

    用于比特映射的 MQSABRE 算法。

    该比特映射算法会考虑量子芯片上的 cnot 门的错误率和执行时间。

    参数：
        - **circuit** (:class:`~.core.circuit.Circuit`) - 需要做比特映射的量子线路。当前仅支持单比特或者两比特量子门，且控制为包含在其中。
        - **topology** (:class:`~.device.QubitsTopology`) - 量子硬件的比特拓扑结构。当前仅支持联通图。
        - **cnoterrorandlength** (List[Tuple[Tuple[int, int], List[float]]]) - CNOT 门的错误率和执行时长。在这里，前两个整数表示拓扑结构中的比特序号。后面由浮点数构成的数组包含两个元素，第一个元素为 CNOT 门的执行错误率，第二个元素为 CNOT 门的执行时长。

    .. py:method:: solve(w: float, alpha1: float, alpha2: float, alpha3: float)

        利用 SABRE 算法来求解比特映射问题。

        参数：
            - **w** (float) - w 参数。更多信息，请参考论文。
            - **alpha1** (float) - alpha1 参数。更多信息，请参考论文。
            - **alpha2** (float) - alpha2 参数。更多信息，请参考论文。
            - **alpha3** (float) - alpha3 参数。更多信息，请参考论文。

        返回：
            Tuple[:class:`~.core.circuit.Circuit`, List[int], List[int]]，一个可以在硬件上执行的量子线路，初始的映射顺序，最后的映射顺序。
