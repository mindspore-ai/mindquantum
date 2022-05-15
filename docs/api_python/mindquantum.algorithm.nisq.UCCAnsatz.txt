Class mindquantum.algorithm.nisq.UCCAnsatz(n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1)

    用于分子模拟的统一耦合簇。

    .. math::

        U(\vec{\theta}) = \prod_{j=1}^{N(N\ge1)}{\prod_{i=0}^{N_{j}}{\exp{(\theta_{i}\hat{\tau}_{i})}}}

    其中 :math:'\帽子{\陶}'是反赫米特运算符。

    注:
        目前，该电路是使用JW变换构建的。
        此外，参考状态波函数（Hartree-Fock）将不包括在内。

    参数：
        n_qubits(int): 量子比特（自旋轨道）的数量。默认值：None。
        n_electrons(int): 电子的数量（占用的自旋轨道）。默认值：无。
        occ_orb(list): 手动分配的占用空间轨道的指数，仅适用于asatz构造。默认值：None。
        vir_orb(list): 手动分配的虚拟空间轨道的指数，仅适用于asatz构造。默认值：None。
        generalized(bool): 是否使用不区分占用轨道或虚拟轨道的广义激励（UGCGSD）。默认值：False。
        trotter_step(int): 特罗特化步骤的顺序。默认值：1。

    样例:
        >>> from mindquantum.algorithm.nisq.chem import UCCAnsatz
        >>> ucc = UCCAnsatz(12, 4, occ_orb=[1],
        ...                 vir_orb=[2, 3],
        ...                 generalized=True,
        ...                 trotter_step=2)
        >>> circuit = ucc.circuit.remove_barrier()
        >>> len(circuit)
        3624
        >>> params_list = ucc.circuit.params_name
        >>> len(params_list)
        48
        >>> circuit[-10:]
        q5: ──●────RX(7π/2)───────H───────●────────────────────────────●───────H──────
              │                           │                            │
        q7: ──X───────H────────RX(π/2)────X────RZ(-0.5*t_1_d0_d_17)────X────RX(7π/2)──
       