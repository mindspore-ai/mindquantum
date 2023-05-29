mindquantum.algorithm.compiler.cxx_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.cxx_decompose(gate: gates.XGate)

    分解一个受控的 `toffoli` 门。

    参数：
        - **gate** (:class:`~.core.gates.XGate`) - 有两个个控制位的 :class:`~.core.gates.XGate` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
