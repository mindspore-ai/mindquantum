mindquantum.framework.MQN2Layer
===============================

.. py:class:: mindquantum.framework.MQN2Layer(expectation_with_grad, weight='normal')

    包含encoder线路和ansatz线路的量子神经网络，encoder线路将经典数据编码成量子态，ansatz线路的参数是可训练的参数，网络输出测量值的模方。

    参数：
        - **expectation_with_grad** (GradOpsWrapper) - 梯度算子，接收encoder数据和ansatz数据，并返回期望值的绝对值和参数相对于期望的梯度值的平方。
        - **weight** (Union[Tensor, str, Initializer, numbers.Number]) - 卷积核的初始化器。它可以是Tensor、字符串、Initializer或数字。指定字符串时，可以使用 ``'TruncatedNormal'``、 ``'Normal'``、 ``'Uniform'``、 ``'HeUniform'`` 和 ``'XavierUniform'`` 分布以及常量'One'和'Zero'分布中的值。支持别名 ``'xavier_uniform'``、 ``'he_uniform'``、 ``'ones'`` 和 ``'zeros'``。同时支持大写和小写。有关更多详细信息，请参阅Initializer的值。默认值： ``'normal'``。

    输入：
        - **enc_data** (Tensor) - encoder数据，即要编码为量子态的Tensor。

    输出：
        Tensor，hamiltonian期望绝对值的平方。

    异常：
        - **ValueError** - 如果 `weight` 的shape长度不等于1，并且 `weight` 的shape[0]不等于 `weight_size`。
