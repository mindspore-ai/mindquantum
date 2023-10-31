mindquantum.algorithm.compiler.cswap_decompose
==============================================

.. py:function:: mindquantum.algorithm.compiler.cswap_decompose(gate: gates.SWAPGate)

    分解一个受控的 :class:`~.core.gates.SWAPGate` 门。

    参数：
        - **gate** (:class:`~.core.gates.SWAPGate`) - 有一个控制位的 :class:`~.core.gates.SWAPGate` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
