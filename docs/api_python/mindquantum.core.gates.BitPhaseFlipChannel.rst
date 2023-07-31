mindquantum.core.gates.BitPhaseFlipChannel
===========================================

.. py:class:: mindquantum.core.gates.BitPhaseFlipChannel(p: float, **kwargs)

    比特相位翻转信道。描述的噪声体现为：以 :math:`P` 的概率翻转比特的量子态和相位（作用 :math:`Y` 门），或以 :math:`1-P` 的概率保持不变（作用 :math:`I` 门）。

    比特相位翻转通道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Y \rho Y

    其中， :math:`\rho` 是密度矩阵形式的量子态； :math:`P` 是作用额外 :math:`Y` 门的概率。

    参数：
        - **p** (int, float) - 发生错误的概率。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了该噪声信道的Kraus算符。
