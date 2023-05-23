mindquantum.algorithm.compiler.cz_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.cz_decompose(gate: gates.ZGate)

    分解一个受控的 :class:`mindquantum.core.gates.ZGate` 门。

    参数：
        - **gate** (:class:`mindquantum.core.gates.ZGate`) - 有一个控制位的 :class:`mindquantum.core.gates.ZGate` 门。

    返回：
        List[:class:`mindquantum.core.circuit.Circuit`]，可能的分解方式。
