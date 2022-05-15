Class mindquantum.simulator.GradOpsWrapper(grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker)

    使用生成此梯度运算符的信息包装梯度运算符。

    参数:
        grad_ops (Union[FunctionType, MethodType])): 返回正向值和梯度w.r.t参数的函数或方法。
        hams (Hamiltonian): 产生这个研究生行动的汉密尔顿式。
        circ_right (Circuit): 生成此梯度操作的右电路。
        circ_left (Circuit): 生成此梯度操作的左电路。
        encoder_params_name (list[str]): 编码器参数名称。
        ansatz_params_name (list[str]): 安萨茨参数名称。
        parallel_worker (int): 运行批处理的并行工作器的数量。
    