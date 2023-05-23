mindquantum.algorithm.compiler.cryy_decompose
=============================================

.. py:function:: mindquantum.algorithm.compiler.cryy_decompose(gate: gates.Ryy)

    分解一个受控的 :class:`mindquantum.core.gates.Ryy` 门。

    参数：
        - **gate** (:class:`mindquantum.core.gates.Ryy`) - 有一个控制位的 :class:`mindquantum.core.gates.Ryy` 门。

    返回：
        List[:class:`mindquantum.core.circuit.Circuit`]，可能的分解方式。
