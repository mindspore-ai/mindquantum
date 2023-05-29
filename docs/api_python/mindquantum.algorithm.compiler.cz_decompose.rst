mindquantum.algorithm.compiler.cz_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.cz_decompose(gate: gates.ZGate)

    分解一个受控的 :class:`~.core.gates.ZGate` 门。

    参数：
        - **gate** (:class:`~.core.gates.ZGate`) - 有一个控制位的 :class:`~.core.gates.ZGate` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
