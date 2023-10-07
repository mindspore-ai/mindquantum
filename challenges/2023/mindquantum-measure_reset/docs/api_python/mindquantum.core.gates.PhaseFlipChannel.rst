mindquantum.core.gates.PhaseFlipChannel
========================================

.. py:class:: mindquantum.core.gates.PhaseFlipChannel(p: float, **kwargs)

    相位翻转信道。描述的噪声体现为：以 :math:`P` 的概率翻转量子比特的相位（应用Z门），或以 :math:`1-P` 的概率保持不变（作用I门）。

    相位翻转信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Z \rho Z

    其中，:math:`\rho` 是密度矩阵形式的量子态； :math:`P` 是作用额外Z门的概率。

    参数：
        - **p** (int, float) - 发生错误的概率。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了该噪声信道的Kraus算符。
