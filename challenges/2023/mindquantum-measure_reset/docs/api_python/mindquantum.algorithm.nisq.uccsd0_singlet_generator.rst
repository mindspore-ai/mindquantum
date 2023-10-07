mindquantum.algorithm.nisq.uccsd0_singlet_generator
====================================================

.. py:function:: mindquantum.algorithm.nisq.uccsd0_singlet_generator(n_qubits=None, n_electrons=None, anti_hermitian=True, occ_orb=None, vir_orb=None, generalized=False)

    利用CCD0 ansatz来生成分子系统的UCCSD算子。

    .. note::
        手动配置的 `occ_orb` 或者 `vir_orb` 会被认为是空间轨道而不是自选轨道，并且会重新改写 `n_electrons` 和 `n_qubits`。 这在某种程度上与活动空间相似，因此可以减少变分参数的数量。但是，它可能不会减少所需要的比特数，因为费米子激发算符是非局部的，例如， :math:`a_{7}^{\dagger} a_{0}` 不仅涉及第0和第7个量子比特，而且还涉及第1个直到第6个量子比特。

    参数：
        - **n_qubits** (int) - 量子比特个数（自旋轨道）。默认值： ``None``。
        - **n_electrons** (int) - 电子个数（占据的自旋轨道）。默认值： ``None``。
        - **anti_hermitian** (bool) - 是否减去该算符的厄米共轭以形成反厄米共轭算符。默认值： ``True``。
        - **occ_orb** (list) - 手动分配的占据空间轨道的序号。默认值： ``None``。
        - **vir_orb** (list) - 手动分配的虚空间轨道的序号。默认值： ``None``。
        - **generalized** (bool) - 是否使用不区分占据轨道或虚轨道的广义激发（UCCGSD）。默认值： ``False``。

    返回：
        FermionOperator，使用CCD0 ansatz生成的UCCSD算子。
