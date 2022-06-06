.. py:class:: mindquantum.core.gates.DepolarizingChannel(p: float, **kwargs)

    量子信道可以描述量子计算中的非相干噪声。

    去极化通道描述的噪声体现为：以P的概率将量子态转变为最大混态（随机作用泡利门（X,Y,Z）的其中一个，每个泡利门的概率都是P/3），或以1-P的概率保持不变（作用I门）。

    去极化信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P/3( X \rho X + Y \rho Y + Z \rho Z)

    其中，ρ是密度矩阵形式的量子态；P是发生去极化错误的概率。

    **参数：**

    - **p** (int, float) - 发生错误的概率。
