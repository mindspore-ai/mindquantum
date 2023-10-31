mindquantum.algorithm.compiler.try_merge
========================================

.. py:function:: mindquantum.algorithm.compiler.try_merge(father_node: GateNode, child_node: GateNode)

    尝试将两个节点融合起来。

    通过本方法，我们将两个互为共轭的量子门融合成一个单位门，或者将两个同类的参数门融合到一起。

    参数：
        - **father_node** (:class:`~.algorithm.compiler.GateNode`) - 想要融合的父 DAG 节点。
        - **child_node** (:class:`~.algorithm.compiler.GateNode`) - 想要融合的子 DAG 节点。

    返回：
        - bool，是否融合成功。
        - List[:class:`~.algorithm.compiler.GateNode`]，融合之后的父节点.
        - :class:`~.core.gates.GlobalPhase`，合并两个给定节点后的global phase gate。
