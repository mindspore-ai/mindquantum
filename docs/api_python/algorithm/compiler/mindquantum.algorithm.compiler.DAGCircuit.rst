mindquantum.algorithm.compiler.DAGCircuit
=========================================

.. py:class:: mindquantum.algorithm.compiler.DAGCircuit(circuit: Circuit)

    量子线路的有向无环图表示（Directed acyclic graph， DAG）。

    参数：
        - **circuit** (:class:`~.core.circuit.Circuit`) - 输入的量子线路。


    .. py:method:: append_node(node: DAGNode)

        添加一个量子门 DAG 节点。

        参数：
            - **node** (:class:`~.algorithm.compiler.DAGNode`) - 想要添加的 DAG 节点。

    .. py:method:: depth()

        返回量子线路的层数。

    .. py:method:: find_all_gate_node()

        查找 :class:`~.algorithm.compiler.DAGCircuit` 图中的所有量子门节点。

        返回：
            List[:class:`~.algorithm.compiler.GateNode`]，:class:`~.algorithm.compiler.DAGCircuit` 中所有 :class:`~.algorithm.compiler.GateNode` 的列表。

    .. py:method:: layering()

        将量子线路进行分层。

        返回：
            List[:class:`~.core.circuit.Circuit`]，分层后的量子线路列表。

    .. py:method:: replace_node_with_dag_circuit(node: DAGNode, coming: "DAGCircuit")
        :abstractmethod:

        用一个 DAG 图来替换给定的节点。

        参数：
            - **node** (:class:`~.algorithm.compiler.DAGNode`) - 原始的节点。
            - **coming** (:class:`~.algorithm.compiler.DAGCircuit`) - 新的 DAG 图。

    .. py:method:: to_circuit()

        将 :class:`~.algorithm.compiler.DAGCircuit` 转化为量子线路。

        返回：
            :class:`~.core.circuit.Circuit` ， DAG 图对应的量子线路。
