.. py:method:: mindquantum.algorithm.nisq.Transform.jordan_wigner()

    应用Jordan-Wigner变换。Jordan-Wigner变换能够保留初始占据数的局域性，并安装如下的形式将费米子转化为玻色子。

    .. math::

        a^\dagger_{j}\rightarrow \sigma^{-}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i}

        a_{j}\rightarrow \sigma^{+}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i},

    其中 :math:`\sigma_{+}=\sigma^{X}+i\sigma^{Y}` 和 :math:`\sigma^{X} = \sigma^{X} - i\sigma^{Y}` 分别是自旋生算符和降算符。

    **返回：**

    QubitOperator，Jordan-Wigner变换后的玻色子算符。
