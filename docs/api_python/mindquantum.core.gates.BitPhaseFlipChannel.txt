Class mindquantum.core.gates.BitPhaseFlipChannel(p: float, **kwargs)

    量子信道，表达量子计算中的非相干噪声。

    位相位翻转通道表示误差，即随机翻转量子位的状态和相位（应用Y门）与概率P，或注意（应用I门）与概率1-P。

    位相位翻转通道应用噪声如下：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Y \rho Y

    其中，γ是密度矩阵类型的量子态；P是应用额外Y门的概率。

    参数:
        p (int, float): 发生错误的概率。

    样例:
        >>> from mindquantum.core.gates import BitPhaseFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += BitPhaseFlipChannel(0.02).on(0)
        >>> circ += BitPhaseFlipChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──BPFC─────●────
                       │
        q1: ──────────BPFC──
       