mindquantum.core.gates.ThermalRelaxationChannel
================================================

.. py:class:: mindquantum.core.gates.ThermalRelaxationChannel(t1: float, t2: float, gate_time: float, **kwargs)

    热弛豫信道。

    热弛豫信道描述了作用量子门时量子比特发生的热退相干和去相位，由 T1、T2 和量子门作用时长决定。

    该信道的Choi矩阵表示如下：

    .. math::

        \begin{gather*}
            \epsilon(\rho) = \text{tr}_1 \left[ \Lambda \left( \rho^T \otimes I \right) \right],
            \Lambda=\begin{pmatrix}
                \epsilon_{T_1} & 0 & 0 & \epsilon_{T_2} \\
                0 & 1-\epsilon_{T_1} & 0 & 0            \\
                0 & 0 & 0 & 0                           \\
                \epsilon_{T_2} & 0 & 0 & 1
            \end{pmatrix}
            \\
            \text{其中}\ \epsilon_{T_1}=e^{-T_g/T_1}, \epsilon_{T_2}=e^{-T_g/T_2}
        \end{gather*}

    这里 :math:`\rho` 是密度矩阵形式的量子态；:math:`\Lambda` 是Choi矩阵，:math:`T_1` 是量子比特的热弛豫时间，:math:`T_2` 是量子比特的相位弛豫时间，:math:`T_g` 是量子门的作用时间。

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
