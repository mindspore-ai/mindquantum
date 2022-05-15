mindquantum.core.circuit.controlled(circuit_fn)

    在量子电路或量子算子上添加控制量子位（可以生成量子电路的函数）

    参数:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): 量子电路，或可以生成量子电路的函数。

    异常:
        TypeError: 电路_fn不是电路或无法返回电路。

    返回:
        可以生成电路的函数。

    样例:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import controlled
        >>> u1 = qft([0, 1])
        >>> u2 = controlled(u1)
        >>> u3 = controlled(qft)
        >>> u3 = u3(2, [0, 1])
        >>> u2(2)
        q0: ──H────PS(π/2)─────────@──
              │       │            │
        q1: ──┼───────●───────H────@──
              │       │       │    │
        q2: ──●───────●───────●────●──
        >>> u3
        q0: ──H────PS(π/2)─────────@──
              │       │            │
        q1: ──┼───────●───────H────@──
              │       │       │    │
        q2: ──●───────●───────●────●──
       