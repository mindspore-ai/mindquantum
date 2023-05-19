mindquantum.algorithm.compiler.ct_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.ct_decompose(gate: gates.TGate)

    分解一个受控的 :class:`TGate` 门。

    参数：
        - **gate** (:class:`TGate`) - 有一个控制位的 :class:`TGate` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
