src.layer
============

适用于MindSpore的算子和cell。

|

.. py:class:: src.layer.MBEOps(loss)

    MBE-VQO算子。通过参数化量子线路 (PQC) 及一系列对量子态的哈密顿期望计算获得损失值。此算子只能在 `PYNATIVE_MODE` 下执行。

    参数：
        - **loss** (src.mbe_loss.MBELoss) - MBE-VQO的损失函数。

    输入：
        - **x** (Tensor) - shape为 :math:`N` 的Tensor，用于ansatz电路，其中 :math:`N` 表示ansatz参数的数量。

    输出：
        Tensor，MBE-VQO求得的损失值。

|

.. py:class:: src.layer.MBELayer(loss, weight='normal')

    包含ansatz线路的量子神经网络，ansatz线路的参数是可训练的参数。

    参数：
        - **loss** (src.mbe_loss.MBELoss) - MBE-VQO的损失函数。
        - **weight** (Union[Tensor, str, Initializer, numbers.Number]) - 量子线路参数的初始化器。它可以是Tensor、字符串、Initializer或数字。指定字符串时，可以使用'TruncatedNormal', 'Normal', 'Uniform', 'HeUniform' 和 'XavierUniform'分布以及常量'One'和'Zero'分布中的值。别名'xavier_uniform'，'he_uniform'，'ones'和'zeros'是可以接受的。大写和小写都可以接受。有关更多详细信息，请参阅Initializer的值。默认值：'normal'。

    输出：
        Tensor，MBE-VQO求得的损失值。

    异常：
        - **ValueError** - 如果 `weight` 的shape长度不等于1，并且 `weight` 的shape[0]不等于 `weight_size`。
