mindquantum.algorithm.compiler.crz_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.crz_decompose(gate: gates.RZ)

    分解一个受控的 :class:`~.core.gates.RZ` 门。

    参数：
        - **gate** (:class:`~.core.gates.RZ`) - 有一个控制位的 :class:`~.core.gates.RZ` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
