mindquantum.core.operators.up_index(index)

    索引顺序，默认情况下，我们将无自旋轨道设置为偶数-奇数-偶数-奇数（0,1,2,3,...）。
    索引偶数的自旋轨道（α轨道）。

    参数:
        index (int): 空间轨道索引。

    返回:
        一个整数，它是关联的自旋轨道的索引。

    样例:
        >>> from mindquantum.core.operators import up_index
        >>> up_index(1)
        2
    