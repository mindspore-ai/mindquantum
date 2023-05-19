mindquantum.algorithm.compiler.ch_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.ch_decompose(gate: gates.HGate)

    分解一个受控的 :class:`HGate` 门。

    参数：
        - **gate** (:class:`HGate`) - 有一个控制位的 :class:`HGate` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
