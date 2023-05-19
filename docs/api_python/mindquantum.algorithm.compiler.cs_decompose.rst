mindquantum.algorithm.compiler.cs_decompose
===========================================

.. py:function:: mindquantum.algorithm.compiler.cs_decompose(gate: gates.SGate)

    分解一个受控的 :class:`SGate` 门。

    参数：
        - **gate** (:class:`SGate`) - 有一个控制位的 :class:`SGate` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
