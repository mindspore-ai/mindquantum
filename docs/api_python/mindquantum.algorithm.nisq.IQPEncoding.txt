Class mindquantum.algorithm.nisq.IQPEncoding(n_feature, first_rotation_gate=<class 'mindquantum.core.gates.basicgate.RZ'>, second_rotation_gate=<class 'mindquantum.core.gates.basicgate.RZ'>, num_repeats=1)

    通用IQP编码。

    参数:
        n_feature (int): 要使用iqp编码编码编码的数据特征数。
        first_rotation_gate (ParamaterGate): 旋转门RX、RY或RZ之一。
        second_rotation_gate (ParamaterGate): 旋转门RX、RY或RZ之一。
        num_repeats (int): 编码迭代次数。

    样例:
        >>> from mindquantum.algorithm.library import IQPEncoding
        >>> iqp = IQPEncoding(3)
        >>> iqp
        q0: ──H────RZ(alpha0)────●───────────────────────────●───────────────────────────────────
                                 │                           │
        q1: ──H────RZ(alpha1)────X────RZ(alpha0 * alpha1)────X────●───────────────────────────●──
                                                                  │                           │
        q2: ──H────RZ(alpha2)─────────────────────────────────────X────RZ(alpha1 * alpha2)────X──
        >>> iqp.circuit.params_name
        ['alpha0', 'alpha1', 'alpha2', 'alpha0 * alpha1', 'alpha1 * alpha2']
        >>> iqp.circuit.params_name
        >>> a = np.array([0, 1, 2])
        >>> iqp.data_preparation(a)
        array([0, 1, 2, 0, 2])
        >>> iqp.circuit.get_qs(pr=iqp.data_preparation(a))
        array([-0.28324704-0.21159186j, -0.28324704-0.21159186j,
                0.31027229+0.16950252j,  0.31027229+0.16950252j,
                0.02500938+0.35266773j,  0.02500938+0.35266773j,
                0.31027229+0.16950252j,  0.31027229+0.16950252j])
       