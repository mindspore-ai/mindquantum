.. py:class:: mindquantum.core.gates.BitPhaseFlipChannel(p: float, **kwargs)

    量子信道可以描述量子计算中的非相干噪声。

    比特相位翻转信道描述的噪声体现为：以P的概率翻转比特的量子态和相位（作用Y门），或以1-P的概率保持不变（作用I门）。

    比特相位翻转通道的数学表示如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Y \rho Y

    其中，ρ是密度矩阵形式的量子态；P是作用额外Y门的概率。

    **参数：**
    - **p** (int, float) - 发生错误的概率。

    **样例：**
        >>> from mindquantum.core.gates import BitPhaseFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += BitPhaseFlipChannel(0.02).on(0)
        >>> circ += BitPhaseFlipChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──BPFC─────●────
                       │
        q1: ──────────BPFC──
       