mindquantum.core.circuit.decompose_single_term_time_evolution(term, para)

    将时间进化门分解为基本量子门。

    这个函数只适用于只有一个保利词的汉密尔顿语。
    例如，exp(-i * t * ham), 火腿只能是一个保利词，如ham = X0 x Y1 x Z2, 此时，术语将是((0, 'X'), (1, 'Y'), (2, 'Z'))。
    当进化时间表示为t = a*x + b*y时，参数将为{'x':a, 'y':b}。

    参数:
        term (tuple, QubitOperator): 仅进化量子比特运算符的汉密尔顿术语。
        para (Union[dict, numbers.Number]): 进化运算符的参数。

    返回:
        电路，量子电路。

    异常:
        ValueError: 如果术语有多个pauli字符串。
        TypeError: 如果术语不是映射。

    样例:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.circuit import decompose_single_term_time_evolution
        >>> ham = QubitOperator('X0 Y1')
        >>> circuit = decompose_single_term_time_evolution(ham, {'a':1})
        >>> print(circuit)
        q0: ─────H───────●───────────────●───────H──────
                         │               │
        q1: ──RX(π/2)────X────RZ(2*a)────X────RX(7π/2)──
       