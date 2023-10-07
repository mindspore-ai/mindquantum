mindquantum.algorithm.compiler.SimpleNeighborCanceler
=====================================================

.. py:class:: mindquantum.algorithm.compiler.SimpleNeighborCanceler()

    如果可能，该编译规则会融合临近的量子门。

    .. py:method:: do(dag_circuit: DAGCircuit)

        原位的将该编译规则运用到 :class:`~.algorithm.compiler.DAGCircuit` 上。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 想要编译的量子线路。
