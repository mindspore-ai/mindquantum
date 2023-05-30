mindquantum.algorithm.nisq.ansatz_variance
==========================================

.. py:function:: mindquantum.algorithm.nisq.ansatz_variance(ansatz: Circuit, ham: Hamiltonian, focus: str, var_range: typing.Tuple[float, float] = (0, np.pi * 2), other_var: np.array = None, atol: float = 0.1, init_batch: int = 20, sim: typing.Union[Simulator, str] = 'mqvector')

    计算变分量子线路中的某个参数的梯度的方差。

    参数：
        - **ansatz** (:class:`~.core.circuit.Circuit`) - 输入的变分量子线路。
        - **ham** (:class:`~.core.operators.Hamiltonian`) - 输入的可观察量哈密顿量。
        - **focus** (str) - 需要检查哪个参数。
        - **var_range** (Tuple[float, float]) - 参数的随机变化范围。默认值： ``(0, 2*np.pi)``。
        - **other_var** (numpy.array) - 其他变量的数值。如果为 ``None``，则每次采样是都是随机数。默认值： ``None``。
        - **atol** (float) - 方差浮动的容忍度。默认值 ``0.1``。
        - **init_batch** (int) - 初始采样时样本点的个数。默认值： ``20``。
        - **sim** (Union[:class:`~.simulator.Simulator`, str]) - 用哪种模拟器来完成任务。默认值： ``mqvector``。
