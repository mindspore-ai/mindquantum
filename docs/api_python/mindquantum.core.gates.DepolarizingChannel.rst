mindquantum.core.gates.DepolarizingChannel
===========================================

.. py:class:: mindquantum.core.gates.DepolarizingChannel(p: float, **kwargs)

    去极化信道。描述的噪声体现为：以 :math:`P` 的概率将量子态转变为最大混态（随机作用泡利门（I、X、Y、Z）的其中一个，每个泡利门的概率都是 :math:`P/4` ），或以 :math:`1-P` 的概率保持不变。

    对于单比特情况，去极化信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P/4( I \rho I + X \rho X + Y \rho Y + Z \rho Z)

    其中，:math:`\rho` 是密度矩阵形式的量子态；:math:`P` 是发生去极化错误的概率。

    该信道还支持作用于多个目标比特。在 :math:`N` 比特情况下，去极化信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + \frac{P}{4^N} \sum_j U_j \rho U_j

    其中，:math:`N` 是目标比特数； :math:`U_j \in \left\{ I, X, Y, Z \right\} ^{\otimes N}` 多比特泡利算符。

    参数：
        - **p** (int, float) - 发生错误的概率。
