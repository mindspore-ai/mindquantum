mindquantum.algorithm.compiler.cxx_decompose
============================================

.. py:function:: mindquantum.algorithm.compiler.cxx_decompose(gate: gates.XGate)

    分解一个受控的 `toffoli` 门。

    参数：
        - **gate** (:class:`XGate`) - 有两个个控制位的 :class:`XGate` 门。

    返回：
        List[:class:`Circuit`]，可能的分解方式。
