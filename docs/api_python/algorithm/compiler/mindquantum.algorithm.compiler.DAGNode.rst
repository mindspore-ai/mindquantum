mindquantum.algorithm.compiler.DAGNode
======================================

.. py:class:: mindquantum.algorithm.compiler.DAGNode()

    DAG 图中的节点。

    一个 DAG 图节点由局域腿标签、子节点和父节点构成。

    .. py:method:: clean()

        清除节点，将节点中的信息删除。

    .. py:method:: insert_after(other_node: "DAGNode")

        将其他节点插入到本节点后面。

        参数：
            - **other_node** (:class:`~.algorithm.compiler.DAGNode`) - 另外一个 DAG 节点。

    .. py:method:: insert_before(other_node: "DAGNode")

        将其他节点插入到本节点前面。

        参数：
            - **other_node** (:class:`~.algorithm.compiler.DAGNode`) - 另外一个 DAG 节点。
