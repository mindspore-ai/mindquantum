Class mindquantum.core.gates.XGate

    Pauli X门，矩阵为：

    .. math::

        {\rm X}=\begin{pmatrix}0&1\\1&0\end{pmatrix}

    为了简单起见，我们将``X`定义为``XGate()``的实例。有关更多重新定义，请参考下面的功能表。

    注:
        为了简单起见，您可以在保利门上执行电源运算符（目前仅适用于保利门）。规则如下：

        .. math::

            X^\theta = RX(\theta\pi)

    样例:
        >>> from mindquantum.core.gates import X
        >>> x1 = X.on(0)
        >>> cnot = X.on(0, 1)
        >>> print(x1)
        X(0)
        >>> print(cnot)
        X(0 <-: 1)
        >>> x1.matrix()
        array([[0, 1],
               [1, 0]])
        >>> x1**2
        RX(2π)
        >>> (x1**'a').coeff
        {'a': 3.141592653589793}, const: 0.0
        >>> (x1**{'a' : 2}).coeff
        {'a': 6.283185307179586}, const: 0.0
       