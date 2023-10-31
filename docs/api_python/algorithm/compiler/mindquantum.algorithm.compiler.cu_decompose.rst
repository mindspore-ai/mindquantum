mindquantum.algorithm.compiler.cu_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.cu_decompose(gate: QuantumGate, with_barrier: bool = False)

    对任意维度的受控幺正门进行分解。

    该门拥有 :math:`m` 个控制比特和 :math:`n` 个作用比特。
    当迭代的调用函数本身时，:math:`m` 将会逐步减小并保持 :math:`n` 恒定。

    参数：
        - **gate** (:class:`~.core.gates.QuantumGate`) - 量子门实例。
        - **with_barrier** (bool) - 是否在分解时加入 :class:`~.core.gates.BarrierGate`。默认值： ``False``。

    返回：
        :class:`~.core.circuit.Circuit`，由单比特门和CNOT门构成的量子线路。
