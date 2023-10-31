mindquantum.algorithm.compiler.connect_two_node
===============================================

.. py:function:: mindquantum.algorithm.compiler.connect_two_node(father_node: DAGNode, child_node: DAGNode, local_index: int)

    通过局域的腿编号，将两个节点连接起来。

    参数：
        - **father_node** (:class:`~.algorithm.compiler.DAGNode`) - 父 DAG 节点。
        - **child_node** (:class:`~.algorithm.compiler.DAGNode`) - 子 DAG 节点。
        - **local_index** (int) - 想要连接的腿的编号。
