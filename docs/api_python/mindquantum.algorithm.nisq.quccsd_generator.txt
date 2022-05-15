mindquantum.algorithm.nisq.quccsd_generator(n_qubits=None, n_electrons=None, anti_hermitian=True, occ_orb=None, vir_orb=None, generalized=False)

    使用量子位激励运算符生成量子位UCCSD(qUCCSD)安萨兹。

    注:
        目前，实现了无限制版本，即来自同一空间轨道但具有不同自旋的激励将使用不同的变分参数。

    参数:
        n_qubits(int): 量子比特（自旋轨道）的数量。默认值：None。
        n_electrons(int): 电子的数量（占用的自旋轨道）。默认值：None。
        anti_hermitian(bool): 是否减去埃尔米特共轭以形成反埃尔米特运算符。默认值：True。
        occ_orb(list): 手动分配的占用空间轨道的指数。默认值：None。
        vir_orb(list): 手动分配的虚拟空间轨道的指数。默认值：None。
        generalized(bool): 是否使用不区分占用轨道或虚拟轨道的广义激励（qUCCGSD）。默认值：False。

    返回:
        QubitExcitationOperator: qUCCSD运算符的生成器。

    样例:
        >>> from mindquantum.algorithm.nisq.chem import quccsd_generator
        >>> quccsd_generator()
        0
        >>> quccsd_generator(4, 2)
        -1.0*q_s_0 [Q0^ Q2] +
        -1.0*q_s_2 [Q0^ Q3] +
        -1.0*q_d_0 [Q1^ Q0^ Q3 Q2] +
        -1.0*q_s_1 [Q1^ Q2] +
        -1.0*q_s_3 [Q1^ Q3] +
        1.0*q_s_0 [Q2^ Q0] +
        1.0*q_s_1 [Q2^ Q1] +
        1.0*q_s_2 [Q3^ Q0] +
        1.0*q_s_3 [Q3^ Q1] +
        1.0*q_d_0 [Q3^ Q2^ Q1 Q0]
        >>> q_op = quccsd_generator(occ_orb=[0], vir_orb=[1], generalized=True)
        >>> q_qubit_op = q_op.to_qubit_operator()
        >>> print(str(q_qubit_op)[:315])
        0.125*I*q_d_4 + 0.125*I*q_d_7 + 0.125*I*q_d_9 [X0 X1 X2 Y3] +
        0.125*I*q_d_4 - 0.125*I*q_d_7 - 0.125*I*q_d_9 [X0 X1 Y2 X3] +
        0.25*I*q_d_12 + 0.25*I*q_d_5 + 0.5*I*q_s_0 - 0.5*I*q_s_3 [X0 Y1] +
        -0.125*I*q_d_4 + 0.125*I*q_d_7 - 0.125*I*q_d_9 [X0 Y1 X2 X3] +
        0.125*I*q_d_4 + 0.125*I*q_d_7 - 0.125*I*q_d_9 [X0 Y1 Y2 Y3] +
    