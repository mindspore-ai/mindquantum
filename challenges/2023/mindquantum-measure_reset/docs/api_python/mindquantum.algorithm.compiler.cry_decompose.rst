mindquantum.algorithm.compiler.cry_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.cry_decompose(gate: gates.RY)

    分解一个受控的 :class:`~.core.gates.RY` 门。

    参数：
        - **gate** (:class:`~.core.gates.RY`) - 有一个控制位的 :class:`~.core.gates.RY` 门。

    返回：
        List[:class:`~.core.circuit.Circuit`]，可能的分解方式。
