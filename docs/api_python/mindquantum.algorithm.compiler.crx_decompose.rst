mindquantum.algorithm.compiler.crx_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.crx_decompose(gate: gates.RX)

    分解一个受控的 :class:`RX` 门。

    参数：
        - **gate** (:class:`RX`) - 有一个控制位的 :class:`RX` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
