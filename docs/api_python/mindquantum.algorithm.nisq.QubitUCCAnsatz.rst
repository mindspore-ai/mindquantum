Class mindquantum.algorithm.nisq.QubitUCCAnsatz(n_qubits=None, n_electrons=None, occ_orb=None, vir_orb=None, generalized=False, trotter_step=1)

    量子位一元耦合簇（qUCC）的一个变体是一元耦合簇的变体，它使用量子位激励算子而不是费米子激励算子。
    量子位激励算子跨越的Fock空间相当于费米子算子，因此可以使用量子位激励算子近似精确的波函数，而牺牲了更高阶的Trotterization。

    qUCC最大的优点是CNOT门的数量比UCC的原始版本小得多，即使使用3阶或4阶Trotterization。
    此外，尽管变分参数的数量增加，但精度也大大提高。

    注:
        不包括哈特里-福克电路。
        目前，不允许generalized=True，因为理论需要验证。
        参考文献：Yordan S. Yordanov等人。物理版本A, 102, 062612 (2020)

    参数:
        n_qubits (int): 模拟中量子位（自旋轨道）的数量。默认值：None。
        n_electrons (int): 给定分子的电子数。默认值：None。
        occ_orb(list): 手动分配的占用空间轨道的指数。默认值：None。
        vir_orb(list): 手动分配的虚拟空间轨道的指数。默认值：None。
        generalized(bool): 是否使用不区分占用轨道或虚拟轨道的广义激励（qUCCGSD）。
            目前，不允许generalized=True，因为理论需要验证。默认值：False。
        trotter_step (int): 蹄步的数量。默认值为1。建议设置大于等于2的值，以获得较好的精度。默认值：1。

    样例:
        >>> from mindquantum.algorithm.nisq.chem import QubitUCCAnsatz
        >>> QubitUCCAnsatz().n_qubits
        0
        >>> qucc = QubitUCCAnsatz(4, 2, trotter_step=2)
        >>> qucc.circuit[:10]
        q0: ──X──────────●──────────X───────────────────────────────X──────────●──────────X───────
              │          │          │                               │          │          │
        q1: ──┼──────────┼──────────┼────X──────────●──────────X────┼──────────┼──────────┼────X──
              │          │          │    │          │          │    │          │          │    │
        q2: ──●────RY(t_0_q_s_0)────●────●────RY(t_0_q_s_1)────●────┼──────────┼──────────┼────┼──
                                                                    │          │          │    │
        q3: ────────────────────────────────────────────────────────●────RY(t_0_q_s_2)────●────●──
        >>> qucc.n_qubits
        4
        >>> qucc_2 = QubitUCCAnsatz(occ_orb=[0, 1], vir_orb=[2])
        >>> qucc_2.operator_pool
        [-1.0*t_0_q_s_0 [Q0^ Q4] +
        1.0*t_0_q_s_0 [Q4^ Q0] , -1.0*t_0_q_s_1 [Q1^ Q4] +
        1.0*t_0_q_s_1 [Q4^ Q1] , -1.0*t_0_q_s_2 [Q2^ Q4] +
        1.0*t_0_q_s_2 [Q4^ Q2] , -1.0*t_0_q_s_3 [Q3^ Q4] +
        1.0*t_0_q_s_3 [Q4^ Q3] , -1.0*t_0_q_s_4 [Q0^ Q5] +
        1.0*t_0_q_s_4 [Q5^ Q0] , -1.0*t_0_q_s_5 [Q1^ Q5] +
        1.0*t_0_q_s_5 [Q5^ Q1] , -1.0*t_0_q_s_6 [Q2^ Q5] +
        1.0*t_0_q_s_6 [Q5^ Q2] , -1.0*t_0_q_s_7 [Q3^ Q5] +
        1.0*t_0_q_s_7 [Q5^ Q3] , -1.0*t_0_q_d_0 [Q1^ Q0^ Q5 Q4] +
        1.0*t_0_q_d_0 [Q5^ Q4^ Q1 Q0] , -1.0*t_0_q_d_1 [Q2^ Q0^ Q5 Q4] +
        1.0*t_0_q_d_1 [Q5^ Q4^ Q2 Q0] , -1.0*t_0_q_d_2 [Q2^ Q1^ Q5 Q4] +
        1.0*t_0_q_d_2 [Q5^ Q4^ Q2 Q1] , -1.0*t_0_q_d_3 [Q3^ Q0^ Q5 Q4] +
        1.0*t_0_q_d_3 [Q5^ Q4^ Q3 Q0] , -1.0*t_0_q_d_4 [Q3^ Q1^ Q5 Q4] +
        1.0*t_0_q_d_4 [Q5^ Q4^ Q3 Q1] , -1.0*t_0_q_d_5 [Q3^ Q2^ Q5 Q4] +
        1.0*t_0_q_d_5 [Q5^ Q4^ Q3 Q2] ]
       