mindquantum.core.circuit.dagger(circuit_fn)

    获取量子电路或量子算子（可以生成量子电路的函数）的隐士匕首

    参数:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): 量子电路，或可以生成量子电路的函数。

    异常:
        TypeError: 如果电路_fn不是电路或无法返回电路。

    返回:
        电路或可以生成电路的函数。

    样例:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import dagger
        >>> u1 = qft([0, 1])
        >>> u2 = dagger(u1)
        >>> u3 = dagger(qft)
        >>> u3 = u3([0, 1])
        >>> u2
        q0: ──@─────────PS(-π/2)────H──
              │            │
        q1: ──@────H───────●───────────
        >>> u3
        q0: ──@─────────PS(-π/2)────H──
              │            │
        q1: ──@────H───────●───────────
       