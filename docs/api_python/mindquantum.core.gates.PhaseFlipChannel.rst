.. py:class:: mindquantum.core.gates.PhaseFlipChannel(p: float, **kwargs)

    量子信道可以描述量子计算中的非相干噪声。

    相位翻转信道描述的噪声体现为：以P的概率翻转量子比特的相位（应用Z门），或以1-P的概率保持不变（作用I门）。

    相位翻转信道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Z \rho Z

    其中，ρ是密度矩阵形式的量子态；P是作用额外Z门的概率。

    **参数：**

    - **p** (int, float) - 发生错误的概率。
