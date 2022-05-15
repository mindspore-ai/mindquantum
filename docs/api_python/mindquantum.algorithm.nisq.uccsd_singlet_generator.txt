mindquantum.algorithm.nisq.uccsd_singlet_generator(n_qubits, n_electrons, anti_hermitian=True)
为具有n_电子的系统创建单子UCCSD生成器

    此函数为UCCSD生成器生成一个Fermion算子，该生成器设计用于作用于由n_qubits自旋轨道和n_电子组成的单个参考状态，即自旋单态算子，这意味着它保守自旋。

    参数:
        n_qubits(int): 用于表示系统的自旋轨道数，这也对应于非紧凑映射中的量子位数。
        n_electrons(int): 物理系统中电子的数量。
        anti_hermitian(bool): 仅生成普通CCSD运算符而不是统一变量的标志，主要用于测试。

    返回:
        Fermion算子，构建UCCSD波函数的UCCSD算子的生成器。

    样例:
        >>> from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator
        >>> uccsd_singlet_generator(4, 2)
        -s_0 [0^ 2] +
        -d1_0 [0^ 2 1^ 3] +
        -s_0 [1^ 3] +
        -d1_0 [1^ 3 0^ 2] +
        s_0 [2^ 0] +
        d1_0 [2^ 0 3^ 1] +
        s_0 [3^ 1] +
        d1_0 [3^ 1 2^ 0]
    