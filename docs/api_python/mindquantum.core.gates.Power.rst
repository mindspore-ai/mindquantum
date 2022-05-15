Class mindquantum.core.gates.Power(gate, t=0.5)

    非参数化门上的功率运算符。

    参数:
        gates (:class:`mindquantum.core.gates.NoneParameterGate`): 应用功率运算符所需的基本门。
        t (int, float): 指数。默认值：0.5。

    样例:
        >>> from mindquantum import Power
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> rx2 = RX(1)
        >>> assert np.all(np.isclose(Power(rx2,0.5).matrix(), rx1.matrix()))
    