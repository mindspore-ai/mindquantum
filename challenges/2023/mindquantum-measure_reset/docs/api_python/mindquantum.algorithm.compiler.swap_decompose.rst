mindquantum.algorithm.compiler.swap_decompose
=============================================

.. py:function:: mindquantum.algorithm.compiler.swap_decompose(gate: gates.SWAPGate)

    分解一个 :class:`~.core.gates.SWAPGate` 门。

    参数：
        - **gate** (:class:`~.core.gates.SWAPGate`) - 一个 :class:`~.core.gates.SWAPGate` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
