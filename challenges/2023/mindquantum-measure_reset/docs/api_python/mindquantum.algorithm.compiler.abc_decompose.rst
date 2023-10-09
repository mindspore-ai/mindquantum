mindquantum.algorithm.compiler.abc_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.abc_decompose(gate: QuantumGate, return_u3: bool = True)

    通过abc分解来分解量子门。

    参数：
        - **gate** (:class:`~.core.gates.QuantumGate`) - 只有一个控制为的单比特量子门。
        - **return_u3** (bool) - 如果为 ``True``，则返回 :class:`~.core.gates.U3` 形式的分解，否则返回 :class:`~.core.gates.UnivMathGate` 形式的分解。默认值： ``True``。

    返回：
        :class:`~.core.circuit.Circuit`，由单比特门和CNOT门构成的量子线路。
