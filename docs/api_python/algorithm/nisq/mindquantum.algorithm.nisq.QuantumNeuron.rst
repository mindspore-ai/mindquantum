mindquantum.algorithm.nisq.QuantumNeuron
=========================================

.. py:class:: mindquantum.algorithm.nisq.QuantumNeuron(weight, gamma=1, bias=0, input_qubits=None, output_qubit=None, ancilla_qubit=None)

    基于RUS(Repeat-Until-Success)策略的量子神经元实现，通过量子电路模拟经典神经元行为和激活函数。

    更多信息请参考 `Quantum neuron: an elementary building block for machine learning on quantum computers <https://arxiv.org/abs/1711.11240>`_。

    工作原理：
        - 使用RUS电路，重复执行量子电路直到获得目标测量结果
        - 测量结果为'0'表示成功应用非线性函数旋转
        - 测量结果为'1'触发恢复操作并重复直到成功

    .. note::
        - 根据经验测试，RUS电路需要至少一次失败（测量结果为'1'）后接一次成功（测量结果为'0'）才能正确应用非线性函数
        - 原始论文中的恢复旋转角度为RY(-π/2)，但基于实验验证，本实现使用RY(π/2)。建议用户在具体应用中仔细验证其行为。

    参数：
        - **weight** (Union[List[float], np.ndarray]) - 权重列表或numpy数组。长度必须等于输入量子比特数，每个权重对应一个输入量子比特。
        - **gamma** (Union[int, float]) - 用于调整权重影响的缩放因子。默认值：``1``。
        - **bias** (Union[int, float]) - 偏置项。默认值：``0``。
        - **input_qubits** (Optional[List[int]]) - 用作输入的量子比特索引列表。如果为None，将使用`[0, 1, ..., len(weight)-1]`。默认值：``None``。
        - **output_qubit** (Optional[int]) - 作为神经元输出的量子比特索引。如果为None，将设置为`ancilla_qubit + 1`。默认值：``None``。
        - **ancilla_qubit** (Optional[int]) - 用于计算的辅助量子比特索引。如果为None，将设置为`len(input_qubits) + 1`。默认值：``None``。

    .. py:method:: circuit
        :property:

        量子神经元的量子电路。

        返回：
            Circuit，量子神经元的量子电路。

    .. py:method:: recovery_circuit
        :property:

        测量结果为'1'时的恢复电路。

        该电路在输出量子比特上应用π/2的Y轴旋转以从不成功的测量结果中恢复。

        .. note::
            原始论文中使用RY(-π/2)旋转，但基于实验验证，本实现使用RY(π/2)。用户应在其具体应用中验证此行为。

        返回：
            Circuit，用于恢复操作的量子电路。
