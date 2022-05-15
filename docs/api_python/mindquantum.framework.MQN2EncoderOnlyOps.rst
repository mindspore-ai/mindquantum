Class mindquantum.framework.MQN2EncoderOnlyOps(expectation_with_grad)

    MindQuantum算子，它得到哈密顿量在量子态上的期望绝对值的平方，由参数化量子电路（PQC）评估。此PQC应仅包含编码器电路。此操作仅受`PYNT_MODE`支持。

    参数:
        expectation_with_grad (GradOpsWrapper): 接收编码器数据和andsatz数据，并返回期望值绝对值和参数梯度值相对于期望的平方。

    输入:
        - **ans_data** (Tensor) - 带形状的张量 :math:`N` 用于andsatz电路，其中 :math:`N` 表示andsatz参数的数量。

    输出:
        张量，汉密尔顿期望值绝对值的平方。

    支持平台:
        ``GPU``, ``CPU``

    样例:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQN2EncoderOnlyOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ, encoder_params_name=circ.params_name)
        >>> data = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> f, g = grad_ops(data)
        >>> np.abs(f) ** 2
        array([[0.00957333],
               [0.07408856]])
        >>> net = MQN2EncoderOnlyOps(grad_ops)
        >>> f_ms = net(ms.Tensor(data))
        >>> f_ms
        Tensor(shape=[2, 1], dtype=Float32, value=
        [[ 9.57333017e-03],
         [ 7.40885586e-02]])
    