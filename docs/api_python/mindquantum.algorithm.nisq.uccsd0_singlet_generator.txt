mindquantum.algorithm.nisq.uccsd0_singlet_generator(n_qubits=None, n_electrons=None, anti_hermitian=True, occ_orb=None, vir_orb=None, generalized=False)

    使用CCD0和分子系统的CCD0生成UCCSD算子。

    注:
        手动分配的occ_orb或vir_orb是空间轨道的索引，而不是自旋轨道。它们将覆盖n_电子和n_qubits。
        这在某种程度上与活动空间相似，因此可以减少变分参数的数量。
        但是，它可能不会减少所需量子位的数量，因为费米子激励算子是非局部的，即 :math:`a_{7}^{\dagger} a_{0}` 不仅涉及第0和第7个量子位，而且还涉及第1、第2个量子位...第6个量子位。

    参数:
        n_qubits(int): 量子位数（自旋轨道）。默认值：None。
        n_electrons(int): 电子的数量（占用的自旋轨道）。默认值：None。
        anti_hermitian(bool): 是否减去埃尔米特共轭以形成反埃尔米特运算符。默认值：True。
        occ_orb(list): 手动分配的占用空间轨道的指数。默认值：None。
        vir_orb(list): 手动分配的虚拟空间轨道的指数。默认值：None。
        generalized(bool): 是否使用不区分占用轨道或虚拟轨道的广义激励（UGCGSD）。默认值：False。

    返回:
        Fermion算子，使用CCD0安萨兹的UCCSD算子的生成器。

    样例:
        >>> from mindquantum.algorithm.nisq.chem.uccsd0 import uccsd0_singlet_generator
        >>> uccsd0_singlet_generator(4, 2)
        -1.0*d0_s_0 [0^ 2] +
        2.0*d0_d_0 [1^ 0^ 3 2] +
        -1.0*d0_s_0 [1^ 3] +
        1.0*d0_s_0 [2^ 0] +
        1.0*d0_s_0 [3^ 1] +
        -2.0*d0_d_0 [3^ 2^ 1 0]
        >>> uccsd0_singlet_generator(4, 2, generalized=True)
        1.0*d0_s_0 - 1.0*d0_s_1 [0^ 2] +
        1.0*d0_d_0 [1^ 0^ 2 1] +
        -1.0*d0_d_0 [1^ 0^ 3 0] +
        -2.0*d0_d_1 [1^ 0^ 3 2] +
        1.0*d0_s_0 - 1.0*d0_s_1 [1^ 3] +
        -1.0*d0_s_0 + 1.0*d0_s_1 [2^ 0] +
        -1.0*d0_d_0 [2^ 1^ 1 0] +
        1.0*d0_d_2 [2^ 1^ 3 2] +
        1.0*d0_d_0 [3^ 0^ 1 0] +
        -1.0*d0_d_2 [3^ 0^ 3 2] +
        -1.0*d0_s_0 + 1.0*d0_s_1 [3^ 1] +
        2.0*d0_d_1 [3^ 2^ 1 0] +
        -1.0*d0_d_2 [3^ 2^ 2 1] +
        1.0*d0_d_2 [3^ 2^ 3 0]
        >>> uccsd0_singlet_generator(6, 2, occ_orb=[0], vir_orb=[1])
        -1.0*d0_s_0 [0^ 2] +
        2.0*d0_d_0 [1^ 0^ 3 2] +
        -1.0*d0_s_0 [1^ 3] +
        1.0*d0_s_0 [2^ 0] +
        1.0*d0_s_0 [3^ 1] +
        -2.0*d0_d_0 [3^ 2^ 1 0]
    