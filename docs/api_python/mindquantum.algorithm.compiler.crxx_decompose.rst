mindquantum.algorithm.compiler.crxx_decompose
=============================================

.. py:function:: mindquantum.algorithm.compiler.crxx_decompose(gate: gates.Rxx)

    分解一个受控的 :class:`Rxx` 门。

    参数：
        - **gate** (:class:`Rxx`) - 有一个控制位的 :class:`Rxx` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
