mindquantum.algorithm.compiler.cy_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.cy_decompose(gate: gates.YGate)

    分解一个受控的 :class:`YGate` 门。

    参数：
        - **gate** (:class:`YGate`) - 有一个控制位的 :class:`YGate` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
