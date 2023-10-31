mindquantum.algorithm.compiler.kak_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.kak_decompose(gate: QuantumGate, return_u3: bool = True)

    通过kak分解来分解任意的两量子比特门。

    更多信息，请参考论文 `An Introduction to Cartan's KAK Decomposition for QC
    Programmers <https://arxiv.org/abs/quant-ph/0406176>`_.

    参数：
        - **gate** (:class:`~.core.gates.QuantumGate`) - 只有一个控制为的单比特量子门。
        - **return_u3** (bool) - 如果为 ``True``，则返回 :class:`~.core.gates.U3` 形式的分解，否则返回 :class:`~.core.gates.UnivMathGate` 形式的分解。默认值： ``True``。

    返回：
        :class:`~.core.circuit.Circuit`，由6个单比特门和最多三个CNOT门构成的量子线路。
