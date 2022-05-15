mindquantum.core.circuit.shift(circ, p)

    移动给定电路的量子位范围。

    参数:
        circ (circuit): 要执行换档运算符的电路。
        p (int): 要移动的量子位距离。

    样例:
        >>> from mindquantum.core.circuit import shift
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().x(1, 0)
        >>> circ
        q0: ──●──
              │
        q1: ──X──
        >>> shift(circ, 1)
        q1: ──●──
              │
        q2: ──X──

    Returns:
        Circuit, the shifted circuit.
       