Class mindquantum.core.circuit.Circuit(gates=None)

    量子电路模块。

    量子电路包含一个或多个量子门，可以在量子模拟器中进行评估。
    通过添加量子门或其他电路，您可以很容易地构建量子电路。

    参数:
        gates (BasicGate, list[BasicGate]): 您可以通过单个量子门或门列表初始化量子电路。门：None。


    样例:
        >>> from mindquantum import Circuit, RX, X
        >>> circuit1 = Circuit()
        >>> circuit1 += RX('a').on(0)
        >>> circuit1 *= 2
        >>> circuit1
        q0: ──RX(a)────RX(a)──
        >>> circuit2 = Circuit([X.on(0,1)])
        >>> circuit3= circuit1 + circuit2
        >>> assert len(circuit3) == 3
        >>> circuit3.summary()
        =======Circuit Summary=======
        |Total number of gates  : 3.|
        |Parameter gates        : 2.|
        |with 1 parameters are  : a.|
        |Number qubit of circuit: 2 |
        =============================
        >>> circuit3
        q0: ──RX(a)────RX(a)────X──
                                │
        q1: ────────────────────●──
       