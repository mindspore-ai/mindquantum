mindquantum.simulator.NoiseBackend
==================================

.. py:class:: mindquantum.simulator.NoiseBackend(base_sim, n_qubits, adder, seed=None, dtype=None)

    基于噪声信道的含噪模拟器。

    参数：
        - **base_sim** (str) - MindQuantum 支持的量子模拟器。
        - **n_qubits** (int) - 该噪声模拟器的比特数。
        - **adder** (:class:`~.core.circuit.ChannelAdderBase`) - 一个信道添加器，可以将一个量子线路转化为噪声线路。
        - **seed** (int) - 一个随机种子。默认值： ``None``。
        - **dtype** (mindquantum.dtype) - 模拟器的数据类型。如果为 ``None``，将选取为 mindquantum.complex128。默认值： ``None``。

    .. py:method:: get_expectation_with_grad(hams, circ_right, circ_left=None, simulator_left=None, parallel_worker=None, pr_shift=False)

        获取一个返回前向值和关于线路参数梯度的函数。

        该方法会先使用当前 ``adder`` 将 `circ_right`（以及可选的 `circ_left`）转换为噪声线路，然后调用底层模拟器计算期望值与梯度。

        参数：
            - **hams** (Union[:class:`~.core.operators.Hamiltonian`, List[:class:`~.core.operators.Hamiltonian`]]) - 需要计算期望的 :class:`~.core.operators.Hamiltonian` 或一组 :class:`~.core.operators.Hamiltonian`。
            - **circ_right** (:class:`~.core.circuit.Circuit`) - 右侧线路，在计算前会自动转化为噪声线路。
            - **circ_left** (:class:`~.core.circuit.Circuit`) - 左侧线路。如果为 ``None``，则默认与 `circ_right` 一致；如果提供，则在计算前会自动转化为噪声线路。默认值： ``None``。
            - **simulator_left** (Union[:class:`~.simulator.NoiseBackend`, :class:`~.simulator.Simulator`]) - 包含左侧量子态的模拟器。如果传入 :class:`~.simulator.NoiseBackend`，将自动使用其底层模拟器。默认值： ``None``。
            - **parallel_worker** (int) - 并行器数量。默认值： ``None``。
            - **pr_shift** (bool) - 是否使用 parameter-shift rule。仅在 mqvector 模拟器中可用，当电路包含噪声信道时该参数将自动启用。默认值： ``False``。

        返回：
            GradOpsWrapper，一个包含生成梯度算子信息的梯度算子包装器。
