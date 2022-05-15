mindquantum.core.operators.sz_operator(n_spatial_orbitals)

    返回sz运算符。

    参数:
        n_spatial_orbitals (int): 空间轨道数(n_qubits // 2)。

    返回:
        FermionOperator，对应于n_spatial_轨道上的sz运算符。

    注:
        默认索引顺序旋转（α）对应于偶数索引，而旋转（β）对应于奇数索引.rpartition()。

    样例:
        >>> from mindquantum.core.operators import sz_operator
        >>> sz_operator(3)
        0.5 [0^ 0] +
        -0.5 [1^ 1] +
        0.5 [2^ 2] +
        -0.5 [3^ 3] +
        0.5 [4^ 4] +
        -0.5 [5^ 5]
    