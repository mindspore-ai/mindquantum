mindquantum.algorithm.compiler.cswap_decompose
==============================================

.. py:function:: mindquantum.algorithm.compiler.cswap_decompose(gate: gates.SWAPGate)

    分解一个受控的 :class:`SWAPGate` 门。

    参数：
        - **gate** (:class:`SWAPGate`) - 有一个控制位的 :class:`SWAPGate` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
