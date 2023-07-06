mindquantum.framework.QRamVecLayer
==================================

.. py:class:: mindquantum.framework.QRamVecLayer(ham, circ, sim, n_thread=None, weight='normal')

    包含qram和ansatz线路的量子神经网络，qram将经典数据直接编码成量子态，ansatz线路的参数是可训练的参数。

    .. note::
        - 对于低于2.0.0版本的MindSpore，不支持将复数张量作为神经网络cell输入，因此我们应该将量子态拆分为实部和虚部，并将其用作输入张量。当MindSpore升级时，这种情况可能会改变。
        - 目前，我们无法计算测量结果相对于每个量子振幅的梯度。

    参数：
        - **ham** (Union[:class:`~.core.operators.Hamiltonian`, List[:class:`~.core.operators.Hamiltonian`]]) - 要想求期望值的哈密顿量或者一组哈密顿量。
        - **circ** (:class:`~.core.circuit.Circuit`) - 变分量子线路。
        - **sim** (:class:`~.simulator.Simulator`) - 做模拟所使用到的模拟器。
        - **n_thread** (int) - 运行一个batch的初始态时的并行数。如果是 ``None``，用单线程来运行。默认值： ``None``。
        - **weight** (Union[Tensor, str, Initializer, numbers.Number]) - 卷积核的初始化器。它可以是Tensor、字符串、Initializer或数字。指定字符串时，可以使用 ``'TruncatedNormal'``、 ``'Normal'``、 ``'Uniform'``、 ``'HeUniform'`` 和 ``'XavierUniform'`` 分布以及常量'One'和'Zero'分布中的值。别名 ``'xavier_uniform'``、 ``'he_uniform'``、 ``'ones'`` 和 ``'zeros'`` 是可以接受的。大写和小写都可以接受。有关更多详细信息，请参阅Initializer的值。默认值： ``'normal'``。

    输入：
        - **qs_r** (Tensor) - 量子态实部的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示全振幅量子态的长度。
        - **qs_i** (Tensor) - 量子态虚部的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示全振幅量子态的长度。

    输出：
        Tensor，hamiltonian的期望值。

    异常：
        - **ValueError** - 如果 `weight` 的shape长度不等于1，并且 `weight` 的shape[0]不等于 `weight_size`。
