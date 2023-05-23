mindquantum.algorithm.compiler.ccx_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.ccx_decompose(gate: gates.XGate)

    分解一个 `toffoli` 门。

    参数：
        - **gate** (:class:`mindquantum.core.gates.XGate`) - 一个有两个控制位的 :class:`mindquantum.core.gates.XGate` 门。

    返回：
        List[:class:`mindquantum.core.circuit.Circuit`]，可能的分解方式。
