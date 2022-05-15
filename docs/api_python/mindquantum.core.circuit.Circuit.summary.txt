mindquantum.core.circuit.Circuit.summary(show=True)

        打印电流电路的信息，包括块号、门号、非参数门号、参数门号和总参数。

        参数:
            show (bool): 是否显示信息。默认值：True。

        样例:
            >>> from mindquantum import Circuit, RX, H
            >>> circuit = Circuit([RX('a').on(1), H.on(1), RX('b').on(0)])
            >>> circuit.summary()
            =========Circuit Summary=========
            |Total number of gates  : 3.    |
            |Parameter gates        : 2.    |
            |with 2 parameters are  : a, b. |
            |Number qubit of circuit: 2     |
            =================================
        