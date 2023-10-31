mindquantum.algorithm.nisq.quccsd_generator
============================================

.. py:function:: mindquantum.algorithm.nisq.quccsd_generator(n_qubits=None, n_electrons=None, anti_hermitian=True, occ_orb=None, vir_orb=None, generalized=False)

    使用比特激发算符生成 `qubit-UCCSD` (qUCCSD) ansatz。

    .. note::
        当前版本为无限制版本，即同一空间轨道但具有不同自旋的激发算符使用不同的变分参数。

    参数：
        - **n_qubits** (int) - 量子比特（自旋轨道）的数量。默认值： ``None``。
        - **n_electrons** (int) - 电子的数量（占据自旋轨道）。默认值： ``None``。
        - **anti_hermitian** (bool) - 是否减去厄米共轭以形成反厄米算符。默认值： ``True``。
        - **occ_orb** (list) - 手动分配的占据空间轨道的序号。默认值： ``None``。
        - **vir_orb** (list) - 手动分配的虚空间轨道的序号。默认值： ``None``。
        - **generalized** (bool) - 是否使用不区分占据轨道和虚轨道的广义激发算符（qUCCGSD）。默认值： ``False``。

    返回：
        QubitExcitationOperator，qUCCSD算符。
