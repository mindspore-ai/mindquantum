mindquantum.algorithm.compiler.cry_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.cry_decompose(gate: gates.RX)

    分解一个受控的 :class:`RY` 门。

    参数：
        - **gate** (:class:`RY`) - 有一个控制位的 :class:`RY` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
