Class mindquantum.framework.MQLayer(expectation_with_grad, weight='normal')

    MindQuantum可训练层。asatz电路的参数是可训练的参数。

    参数:
        expectation_with_grad (GradOpsWrapper): 接收编码器数据和andsatz数据，并返回参数相对于期望的期望值和梯度值的梯度值。
        weight (Union[Tensor, str, Initializer, numbers.Number]): 卷积核的初始化器。它可以是Tensor、字符串、初始化器或数字。
            指定字符串时，可以使用“截断正常”、“正常”、“统一”、“HeUnal”和“XavierUnal”分布以及常量“一”和“零”分布中的值。别名“xavier_unal”、“he_unal”、“1”和“零”是可以接受的。大写和小写都可以接受。有关更多详细信息，请参阅初始化器的值。默认值：“normal”。

    输入:
        - **enc_data** (Tensor) - 要编码为量子状态的编码器数据的Tensor。

    输出:
        张量，汉密尔顿的期望值。

    异常:
        ValueError: 如果`重量`的形状长度不等于1或`重量`的shape[0]，不等于`重量大小`。

    支持平台:
        ``GPU``, ``CPU``

    样例:
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQLayer
        >>> import mindspore as ms
        >>> ms.set_seed(42)
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> enc = Circuit().ry('a', 0)
        >>> ans = Circuit().h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, enc+ans,
        ...                                          encoder_params_name=['a'],
        ...                                          ansatz_params_name=['b'])
        >>> enc_data = ms.Tensor(np.array([[0.1]]))
        >>> net =  MQLayer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net(enc_data)
        >>> net.weight.asnumpy()
        array([3.1423748], dtype=float32)
        >>> net(enc_data)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[-9.98333842e-02]])
    