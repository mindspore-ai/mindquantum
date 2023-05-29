mindquantum.algorithm.compiler.ch_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.ch_decompose(gate: gates.HGate)

    分解一个受控的 :class:`~.core.gates.HGate` 门。

    参数：
        - **gate** (:class:`~.core.gates.HGate`) - 有一个控制位的 :class:`~.core.gates.HGate` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
