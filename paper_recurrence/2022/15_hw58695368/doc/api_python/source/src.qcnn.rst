src.qcnn
=============================

可训练模型。

|

.. py:class:: src.qcnn.QCNNet(n_qubits, encoder)

    QCNN(Quantum Convolutional Neural Network)封装。

    参数：
        - **n_qubits** (int) - 量子比特数。
        - **encoder** (Circuit) - 编码量子线路。

    .. py:method:: train(epoch, train_loader, callbacks)

        模型训练接口。

        参数：
            - **epoch** (int) - 训练执行轮次。
            - **train_loader** Dataset) - 一个训练数据集迭代器。
            - **callbacks** (Optional[list[Callback], Callback]) - 训练过程中需要执行的回调对象或者回调对象列表。

    .. py:method:: export_trained_parameters(checkpoint_name)

        导出模型参数。

        参数：
            - **checkpoint_name** (str) - 模型参数文件名。

    .. py:method:: load_trained_parameters(checkpoint_name)

        导入模型参数。

        参数：
            - **checkpoint_name** (str) - 模型参数文件名。

    .. py:method:: predict(origin_test_x)

        输入样本得到预测结果。

        参数：
            - **origin_test_x** (Union[Tensor, list[Tensor], tuple[Tensor]], 可选) - 预测样本，数据可以是单个张量、张量列表或张量元组。

        返回：
            返回预测结果，类型是Tensor或Tensor元组。
