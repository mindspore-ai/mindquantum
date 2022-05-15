Class mindquantum.core.circuit.U3(a, b, c, obj_qubit=None)

    该电路表示任意单量子位门。

    参数:
        a (Union[numbers.Number, dict, ParameterResolver]): U3电路的第一个参数。
        b (Union[numbers.Number, dict, ParameterResolver]): U3电路的第二个参数。
        c (Union[numbers.Number, dict, ParameterResolver]): U3电路的第三个参数。
        obj_qubit (int): U3电路将作用于哪个量子位。默认值：None。

    样例:
        >>> from mindquantum.core import U3
        >>> U3('a','b','c')
        q0: ──RZ(a)────RX(-π/2)────RZ(b)────RX(π/2)────RZ(c)──
       