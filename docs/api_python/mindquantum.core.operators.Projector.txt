Class mindquantum.core.operators.Projector(proj)

    投影仪操作员。

    对于投影仪，如下所示：

    .. math::
        \left|01\right>\left<01\right|\otimes I^2

    字符串格式为“01II”。

    注:
        下索引量子位位于布拉和ket字符串格式的右端。

    参数:
        proj (str): 投影仪的字符串格式。

    样例:
        >>> from mindquantum.core.operators import Projector
        >>> p = Projector('II010')
        >>> p
        I2 ⊗ ¦010⟩⟨010¦
       