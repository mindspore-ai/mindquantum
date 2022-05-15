mindquantum.core.circuit.change_param_name(circuit_fn, name_map)

    更改参数化量子电路或参数化量子算子（可以生成参数化量子电路的函数）的参数名称。

    参数:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): 量子电路，或可以生成量子电路的函数。
        name_map (dict): 参数名称映射dict。

    异常:
        TypeError: 如果name_map不是映射。
        TypeError: 如果name_map的键不是字符串。
        TypeError: 如果name_map的值不是字符串。
        TypeError: 如果电路_fn不是电路或无法返回电路。

    异常:
        电路或可以生成电路的函数。

    样例:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import change_param_name
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = change_param_name(u1, {'a': 'b'})
        >>> u3 = change_param_name(u, {'a': 'b'})
        >>> u3 = u3(0)
        >>> u2
        q0: ──H────RX(b)──
        >>> u3
        q0: ──H────RX(b)──
       