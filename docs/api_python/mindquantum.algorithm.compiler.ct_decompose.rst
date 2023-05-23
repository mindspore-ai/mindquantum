mindquantum.algorithm.compiler.ct_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.ct_decompose(gate: gates.TGate)

    分解一个受控的 :class:`mindquantum.core.gates.TGate` 门。

    参数：
        - **gate** (:class:`mindquantum.core.gates.TGate`) - 有一个控制位的 :class:`mindquantum.core.gates.TGate` 门。

    返回：
        List[:class:`mindquantum.core.circuit.Circuit`]，可能的分解方式。
