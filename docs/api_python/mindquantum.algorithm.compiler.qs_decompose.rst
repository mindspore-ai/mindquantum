mindquantum.algorithm.compiler.qs_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.qs_decompose(gate: QuantumGate, with_barrier: bool = False)

    任意维幺正量子门的矩阵的香农分解。

    该分解方法中的CNOT门数量为：

    .. math::

        O(4^n)

    了解更多详细信息，请参考 `Synthesis of Quantum Logic Circuits <https://arxiv.org/abs/quant-ph/0406176>`_。

    参数：
        - **gate** (:class:`~.core.gates.QuantumGate`) - 量子门实例。
        - **with_barrier** (bool) - 是否在分解时加入 :class:`~.core.gates.BarrierGate`。默认值： ``False``。

    返回：
        :class:`~.core.circuit.Circuit`，由单比特门和CNOT门构成的量子线路。
