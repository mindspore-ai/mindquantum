mindquantum.core.gates.ThermalRelaxationChannel
================================================

.. py:class:: mindquantum.core.gates.ThermalRelaxationChannel(self, t1: float, t2: float, gate_time: float, **kwargs)

    热弛豫信道。

    热弛豫信道描述了作用量子门时量子比特发生的热退相干和去相位，由 T1、T2 和量子门作用时长决定。

    参数：
        - **t1** (int, float) - 量子比特的T1。
        - **t2** (int, float) - 量子比特的T2。
        - **gate_time** (int, float) - 量子门的作用时长。

    .. py:method:: get_cpp_obj()

        获取底层c++对象。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了该噪声信道的Kraus算符。
