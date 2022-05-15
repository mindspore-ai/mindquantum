Class mindquantum.framework.MQEncoderOnlyOps(expectation_with_grad)

    通过参数化量子电路（PQC）评估的量子态上获得汉密尔顿人的期望的MindQuantum算子。此PQC应仅包含编码器电路。此操作仅受`PYNT_MODE`支持。

    参数:
        expectation_with_grad (GradOpsWrapper): 接收编码器数据和安萨兹数据，并返回参数相对于期望的期望值和梯度值的梯度值。

    输入:
        - **enc_data** (Tensor) - 具有形状的编码器数据的Tensor :math:`(N, M)` ，您希望编码为量子状态，其中 :math:`N` 表示批处理大小和 :math:`M` 表示编码器参数的数量。

    输出:
        张量，汉密尔顿的期望值。

    支持平台:
        ``GPU``, ``CPU``

    样例:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQEncoderOnlyOps
        >>> import mindspore as ms
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ, encoder_params_name=circ.params_name)
        >>> data = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> f, g = grad_ops(data)
        >>> f
        array([[0.0978434 +0.j],
               [0.27219214+0.j]])
        >>> net = MQEncoderOnlyOps(grad_ops)
        >>> f_ms = net(ms.Tensor(data))
        >>> f_ms
        Tensor(shape=[2, 1], dtype=Float32, value=
        [[ 9.78433937e-02],
         [ 2.72192121e-01]])
    