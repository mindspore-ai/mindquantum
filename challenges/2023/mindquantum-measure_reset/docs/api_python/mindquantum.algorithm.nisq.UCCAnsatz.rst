mindquantum.algorithm.nisq.UCCAnsatz
=====================================

.. py:class:: mindquantum.algorithm.nisq.UCCAnsatz(n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1)

    用于分子模拟的幺正耦合簇。

    .. math::

        U(\vec{\theta}) = \prod_{j=1}^{N(N\ge1)}{\prod_{i=0}^{N_{j}}{\exp{(\theta_{i}\hat{\tau}_{i})}}}

    其中，:math:`\hat{\tau}` 是反厄米算符。

    .. note::
        目前，该电路是使用JW变换构建的。
        此外，不包括参考状态波函数（Hartree-Fock）。

    参数：
        - **n_qubits** (int) - 量子比特（自旋轨道）的数量。默认值： ``None``。
        - **n_electrons** (int) - 电子的数量（占用的自旋轨道）。默认值： ``None``。
        - **occ_orb** (list) - 手动分配的占用空间轨道的索引，仅适用于ansatz构造。默认值： ``None``。
        - **vir_orb** (list) - 手动分配的虚拟空间轨道的索引，仅适用于ansatz构造。默认值： ``None``。
        - **generalized** (bool) - 是否使用不区分占用轨道或虚拟轨道的广义激励（UCCGSD）。默认值： ``False``。
        - **trotter_step** (int) - Trotterization的阶数。默认值： ``1``。
