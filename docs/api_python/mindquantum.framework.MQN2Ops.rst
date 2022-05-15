Class mindquantum.framework.MQN2Ops(expectation_with_grad)

    MindQuantum算子，它得到由参数化量子电路（PQC）评估的量子态上哈密顿量期望绝对值的平方。该PQC应包含一个编码器电路和一个andsatz电路。此操作仅受`PYNT_MODE`支持。

    .. math::

        O = \left|\left<\varphi\right| U^\dagger_l H U_r\left|\psi\right>\right|^2

    参数:
        expectation_with_grad (GradOpsWrapper): 接收编码器数据和andsatz数据，并返回期望值绝对值和参数梯度值相对于期望的平方。

    输入:
        - **enc_data** (Tensor) - 具有形状的编码器数据的Tensor :math:`(N, M)` ，您要编码为量子状态，其中 :math:`N` 表示批处理大小和 :math:`M` 表示编码器参数的数量。
        - **ans_data** (Tensor) - 带形状的张量 :math:`N` 用于andsatz电路，其中 :math:`N` 表示andsatz参数的数量。

    输出:
        张量，汉密尔顿期望值绝对值的平方。

    支持平台:
        ``GPU``, ``CPU``

    样例:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2Ops
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0)
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc+ans,
        ...                                          encoder_params_name=['a'],
        ...                                          ansatz_params_name=['b'])
        >>> enc_data = np.array([[0.1]])
        >>> ans_data = np.array([0.2])
        >>> f, g_enc, g_ans = grad_ops(enc_data, ans_data)
        >>> np.abs(f) ** 2
        array([[0.00957333]])
        >>> net = MQN2Ops(grad_ops)
        >>> f_ms = net(ms.Tensor(enc_data), ms.Tensor(ans_data))
        >>> f_ms
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 9.57333017e-03]])
    