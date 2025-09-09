mindquantum.algorithm.compiler.U3Fusion
==========================================

.. py:class:: mindquantum.algorithm.compiler.U3Fusion(rule_name="U3Fusion", log_level=0, with_global_phase=False)

    将连续的单量子比特门融合成一个U3门。

    该规则会扫描线路并将作用在同一个量子比特上的连续单量子比特门组合成一个U3门。对于独立的单量子比特门，也会被转换为U3形式。
    可选择是否跟踪和包含全局相位。

    参数：
        - **rule_name** (str) - 编译规则的名称。默认值："U3Fusion"。
        - **log_level** (int) - 显示日志级别。默认值：0。
        - **with_global_phase** (bool) - 是否包含全局相位门。默认值：False。

    .. py:method:: do(dag_circuit)

        将单量子比特门融合规则应用到量子线路上。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - DAG形式的输入线路。

        返回：
            bool，如果执行了任何融合和转换操作则返回True，否则返回False。 
