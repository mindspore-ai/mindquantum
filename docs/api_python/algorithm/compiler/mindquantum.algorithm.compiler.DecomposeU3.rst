mindquantum.algorithm.compiler.DecomposeU3
===========================================

.. py:class:: mindquantum.algorithm.compiler.DecomposeU3(method='standard')

    将U3门分解为Z-X-Z-X-Z旋转序列。

    分解可以遵循以下两种方法之一:

    1. 标准方法: U3(θ,φ,λ) = Rz(φ)Rx(-π/2)Rz(θ)Rx(π/2)Rz(λ)
    2. 替代方法: U3(θ,φ,λ) = Rz(φ)Rx(π/2)Rz(π-θ)Rx(π/2)Rz(λ-π)

    参数：
        - **method** (str) - 使用的分解方法,可选'standard'或'alternative'。默认值:'standard'。

    .. py:method:: do(dag_circuit)

        执行U3门分解规则。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 量子线路的DAG图。

        返回：
            bool，如果执行了任何分解操作则返回True，否则返回False。
