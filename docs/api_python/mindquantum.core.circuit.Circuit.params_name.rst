mindquantum.core.circuit.Circuit.params_name

        获取此电路的参数名称。

        返回:
            list，包含参数名称的列表。

        样例:
            >>> from mindquantum.core.gates import RX
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit(RX({'a': 1, 'b': 2}).on(0))
            >>> circuit.params_name
            ['a', 'b']
        