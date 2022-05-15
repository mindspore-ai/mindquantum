Class mindquantum.core.gates.BitFlipChannel(p: float, **kwargs)

    量子信道，表达量子计算中的非相干噪声。

    位翻转通道表示错误，即随机翻转量子位（应用X门），概率P，或注意（应用I门），概率1-P。

    位翻转通道应用噪声为：

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P X \rho X

    其中，P是密度矩阵类型的量子态；P是应用额外X门的概率。

    参数:
        p (int, float): 发生错误的概率。

    样例:
        >>> from mindquantum.core.gates import BitFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += BitFlipChannel(0.02).on(0)
        >>> circ += BitFlipChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──BFC─────●───
                      │
        q1: ─────────BFC──
       