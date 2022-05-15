Class mindquantum.core.gates.DepolarizingChannel(p: float, **kwargs)

    量子信道，表达量子计算中的非相干噪声。

    去极化通道通过随机应用具有相同概率P/3的保利门（X,Y,Z）之一，表示具有概率P的误差，将量子比特的量子状态转变为最大混合状态。
    它有概率1-P什么都不改变（应用I门）。

    去极化通道将噪声应用为：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P/3( X \rho X + Y \rho Y + Z \rho Z)

    其中，P是密度矩阵类型的量子态；P是发生去极化误差的概率。

    参数:
        p (int, float): 发生错误的概率。

    样例:
        >>> from mindquantum.core.gates import DepolarizingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += DepolarizingChannel(0.02).on(0)
        >>> circ += DepolarizingChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──DC────●───
                    │
        q1: ────────DC──
       