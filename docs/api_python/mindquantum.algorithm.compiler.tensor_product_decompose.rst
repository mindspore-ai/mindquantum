mindquantum.algorithm.compiler.tensor_product_decompose
=======================================================

.. py:function:: mindquantum.algorithm.compiler.tensor_product_decompose(gate: QuantumGate, return_u3: bool = True)

    量比特量子门的张量直积分解。

    参数：
        - **gate** (:class:`~.core.gates.QuantumGate`) - 一个两比特量子门。
        - **return_u3** (bool) - 如果为 ``True``，则返回 :class:`~.core.gates.U3` 形式的分解，否则返回 :class:`~.core.gates.UnivMathGate` 形式的分解。默认值： ``True``。

    返回：
        :class:`~.core.circuit.Circuit`，包含两个单比特门。
