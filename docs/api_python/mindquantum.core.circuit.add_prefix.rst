mindquantum.core.circuit.add_prefix(circuit_fn, prefix)

    在参数化量子电路或参数化量子算子（可以生成参数化量子电路的函数）的参数上添加前缀。

    参数:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): 量子电路，或可以生成量子电路的函数。
        prefix (str): 要添加到每个参数中的前缀。

    异常:
        TypeError: 如果前缀不是字符串。
        TypeError: 电路_fn不是电路或无法返回电路。

    返回:
        电路或可以生成电路的函数。

    样例:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import add_prefix
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = add_prefix(u1, 'ansatz')
        >>> u3 = add_prefix(u, 'ansatz')
        >>> u3 = u3(0)
        >>> u2
        q0: ──H────RX(ansatz_a)──
        >>> u3
        q0: ──H────RX(ansatz_a)──
       