mindquantum.simulator.GradOpsWrapper
=====================================

.. py:class:: mindquantum.simulator.GradOpsWrapper(grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker, sim=None)

    用生成梯度算子的信息包装梯度算子。

    参数：
        - **grad_ops** (Union[FunctionType, MethodType]) - 返回前向值和线路参数梯度的函数或方法。
        - **hams** (Hamiltonian) - 生成这个梯度算子的hamiltonian。
        - **circ_right** (Circuit) - 生成这个梯度算子的右电路。
        - **circ_left** (Circuit) - 生成这个梯度算子的左电路。
        - **encoder_params_name** (list[str]) - encoder参数名称。
        - **ansatz_params_name** (list[str]) - ansatz参数名称。
        - **parallel_worker** (int) - 运行批处理的并行工作器数量。
        - **sim** (Simulator) - 该梯度算子所使用的模拟器。

    .. py:method:: set_str(grad_str)

        设置梯度算子的表达式。

        参数：
            - **grad_str** (str) - QNN运算符的字符串。
