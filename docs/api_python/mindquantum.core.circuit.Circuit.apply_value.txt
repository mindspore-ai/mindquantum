mindquantum.core.circuit.Circuit.apply_value(pr)

        将此电路转换为具有输入参数的非参数化电路。

        参数:
            pr (Union[dict, ParameterResolver]): 要应用到此电路中的参数。

        返回:
            电路，非参数化电路。

        样例:
            >>> from mindquantum.core.gates import X, RX
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit()
            >>> circuit += X.on(0)
            >>> circuit += RX({'a': 2}).on(0)
            >>> circuit = circuit.apply_value({'a': 1.5})
            >>> circuit
            q0: ──X────RX(3)──
           