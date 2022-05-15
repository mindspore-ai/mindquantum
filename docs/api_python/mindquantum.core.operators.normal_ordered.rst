mindquantum.core.operators.normal_ordered(fermion_operator)

    计算并返回FermionOperator的正常阶数。
    根据惯例，正常排序意味着术语从最高模式索引（左侧）到最低（右侧）排序。
    此外，创建运算符首先出现，然后跟随湮灭运算符。
    例如 3 4^ :math:`\rightarrow` - 4^ 3.

    参数:
        fermion_operator(FermionOperator): 只有费米子类型运算符才有这样的形式。

    返回:
        FermionOperator, 正常有序的费米子算子。

    样例:
        >>> from mindquantum.core.operators import FermionOperator
        >>> from mindquantum.core.operators import normal_ordered
        >>> op = FermionOperator("3 4^", 'a')
        >>> normal_ordered(op)
        -a [4^ 3]
    