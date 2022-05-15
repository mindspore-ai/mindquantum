mindquantum.core.circuit.apply(circuit_fn, qubits)

    将量子电路或量子算子（一种可以生成量子电路的函数）应用于不同的量子比特。

    参数:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): 量子电路，或可以生成量子电路的函数。
        qubits (list[int]): 要应用的新量子位。

    异常:
        TypeError: 如果量子位不是列表。
        ValueError: 如果量子位的任何元素为负数。
        TypeError: 如果电路_fn不是电路或无法返回电路。

    返回:
        电路或可以生成电路的函数。

    样例:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import apply
        >>> u1 = qft([0, 1])
        >>> u2 = apply(u1, [1, 0])
        >>> u3 = apply(qft, [1, 0])
        >>> u3 = u3([0, 1])
        >>> u2
        q0: ──────────●───────H────@──
                      │            │
        q1: ──H────PS(π/2)─────────@──
        >>> u3
        q0: ──────────●───────H────@──
                      │            │
        q1: ──H────PS(π/2)─────────@──
       