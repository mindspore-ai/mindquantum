mindquantum.algorithm.compiler.BasicDecompose
=============================================

.. py:class:: mindquantum.algorithm.compiler.BasicDecompose(prefer_u3=False)

    将量子线路编译成简单的量子门。

    本编译规则可编译多控制的量子门、自定义量子门和多泡利旋转门。

    参数：
        - **prefer_u3** (bool) - 是否优先选择 :class:`~.core.gates.U3` 来进行分解。默认值： ``False``。

    .. py:method:: do(dag_circuit: DAGCircuit)

        原位的将该多控制和自定义量子门编译规则运用到 :class:`~.algorithm.compiler.DAGCircuit` 上。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 量子线路的 DAG 图。
