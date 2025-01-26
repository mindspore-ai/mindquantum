mindquantum.algorithm.compiler.DecomposeU3
===========================================

.. py:class:: mindquantum.algorithm.compiler.DecomposeU3()

    将U3门分解为Z-X-Z-X-Z旋转序列。

    分解遵循以下等式:
    U3(θ,φ,λ) = Rz(φ)Rx(-π/2)Rz(θ)Rx(π/2)Rz(λ)

    .. py:method:: do(dag_circuit: DAGCircuit)

        执行U3门分解规则。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 量子线路的DAG图。

        返回：
            bool，如果执行了任何分解操作则返回True，否则返回False。
