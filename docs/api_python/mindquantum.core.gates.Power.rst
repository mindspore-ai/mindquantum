.. py:class:: mindquantum.core.gates.Power(gate, t=0.5)

    作用在非参数化门上的指数运算符。

    **参数：**
    - **gates** (:class:`mindquantum.core.gates.NoneParameterGate`) - 你要作用指数运算符的基本门。
    - **t** (int, float) - 指数。默认值：0.5。

    **样例：**
        >>> from mindquantum import Power
        >>> import numpy as np
        >>> rx1 = RX(0.5)
        >>> rx2 = RX(1)
        >>> assert np.all(np.isclose(Power(rx2,0.5).matrix(), rx1.matrix()))
    