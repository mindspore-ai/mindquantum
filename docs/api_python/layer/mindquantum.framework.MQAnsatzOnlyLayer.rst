mindquantum.framework.MQAnsatzOnlyLayer
=======================================

.. py:class:: mindquantum.framework.MQAnsatzOnlyLayer(expectation_with_grad, weight='normal')

    仅包含ansatz线路的量子神经网络，ansatz线路的参数是可训练的参数。

    参数：
        - **expectation_with_grad** (GradOpsWrapper) - 梯度算子，接收encoder数据和ansatz数据，并返回期望值和参数相对于期望的梯度值。
        - **weight** (Union[Tensor, str, Initializer, numbers.Number]) - 卷积核的初始化器。它可以是Tensor、字符串、Initializer或数字。指定字符串时，可以使用 ``'TruncatedNormal'``、 ``'Normal'``、 ``'Uniform'``、 ``'HeUniform'`` 和 ``'XavierUniform'`` 分布以及常量'One'和'Zero'分布中的值。支持别名 ``'xavier_uniform'``、 ``'he_uniform'``、 ``'ones'`` 和 ``'zeros'``。同时支持大写和小写。有关更多详细信息，请参阅Initializer的值。默认值： ``'normal'``。

    输出：
        Tensor，hamiltonian的期望值。

    异常：
        - **ValueError** - 如果 `weight` 的shape长度不等于1，并且 `weight` 的shape[0]不等于 `weight_size`。
