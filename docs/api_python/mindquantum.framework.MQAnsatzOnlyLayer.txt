Class mindquantum.framework.MQAnsatzOnlyLayer(expectation_with_grad, weight='normal')

    MindQuantum可训练层。ansatz电路的参数是可训练的参数。

    参数：
        expectation_with_grad (GradOpsWrapper)：接收编码器数据和andsatz数据，并返回参数相对于期望的期望值和梯度值的梯度值。
        weight (Union[Tensor, str, Initializer, numbers.Number])：卷积核的初始化器。它可以是Tensor、字符串、初始化器或数字。
        指定字符串时，可以使用'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' 和 'XavierUniform'分布以及常量'One'和'Zero'分布中的值。别名'xavier_uniform'，'he_uniform'，'ones'和'zeros'是可以接受的。大写和小写都可以接受。有关更多详细信息，请参阅初始化器的值。默认值：'normal'。

    输出：
        Tensor，hamiltonian的期望值。

    异常：
        ValueError：如果`weight`的形状长度不等于1，并且`weight` 的形状[0]不等于`weight_size`。

    支持的平台：
        ``GPU``, ``CPU``

    样例：
        >>> import numpy as np
        >>> from mindquantum import Circuit, Hamiltonian, QubitOperator
        >>> from mindquantum import Simulator, MQAnsatzOnlyLayer
        >>> import mindspore as ms
        >>> ms.set_seed(42)
        >>> ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
        >>> circ = Circuit().ry('a', 0).h(0).rx('b', 0)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> sim = Simulator('projectq', 1)
        >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
        >>> net =  MQAnsatzOnlyLayer(grad_ops)
        >>> opti = ms.nn.Adam(net.trainable_params(), learning_rate=0.1)
        >>> train_net = ms.nn.TrainOneStepCell(net, opti)
        >>> for i in range(100):
        ...     train_net()
        >>> net.weight.asnumpy()
        array([-1.5720805e+00,  1.7390326e-04], dtype=float32)
        >>> net()
        Tensor(shape=[1], dtype=Float32, value= [-9.99999166e-01])
])
