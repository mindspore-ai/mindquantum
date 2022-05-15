Class mindquantum.core.operators.TimeEvolution(ops: mindquantum.core.operators.qubit_operator.QubitOperator, time=None)

    可以生成克罗斯电路的时间进化算子。

    时间进化运算符将执行以下进化：

    .. math::

        \left|\varphi(t)\right>=e^{-iHt}\left|\varphi(0)\right>

    注:
        哈密顿量应该是参数化或非参数化QubitOperator。
        如果QubitOperator有多个术语，则将使用一阶蹄子分解。

    参数:
        ops (QubitOperator): 量子位运算符哈密顿量，可以参数化，也可以非参数化。
        time (Union[numbers.Number, dict, ParameterResolver]): 进化时间，可以是数字或参数解析器。如果无，时间将设置为1。默认值：None。

    样例:
        >>> from mindquantum.core.operators import TimeEvolution, QubitOperator
        >>> q1 = QubitOperator('Z0 Y1', 'a')
        >>> q2 = QubitOperator('X0 Z1', 'b')
        >>> ops1 = q1 + q2
        >>> ops2 = q2 + q1
        >>> TimeEvolution(ops1).circuit
        q0: ─────────────●───────────────●───────H────────●───────────────●────H──
                         │               │                │               │
        q1: ──RX(π/2)────X────RZ(2*a)────X────RX(7π/2)────X────RZ(2*b)────X───────
        >>> TimeEvolution(ops2).circuit
        q0: ──H────●───────────────●───────H───────●───────────────●──────────────
                   │               │               │               │
        q1: ───────X────RZ(2*b)────X────RX(π/2)────X────RZ(2*a)────X────RX(7π/2)──
       