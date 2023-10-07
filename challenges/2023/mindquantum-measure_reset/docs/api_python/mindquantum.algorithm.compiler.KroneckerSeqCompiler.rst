mindquantum.algorithm.compiler.KroneckerSeqCompiler
===================================================

.. py:class:: mindquantum.algorithm.compiler.KroneckerSeqCompiler(compilers: typing.List[BasicCompilerRule], rule_name="KroneckerSeqCompiler", log_level=0)

    正交化编译规则。

    序列中的编译规则会重复执行，直至所有规则不能编译线路中的任何量子门。

    参数：
        - **compilers** (List[:class:`~.algorithm.compiler.BasicCompilerRule`]) - 所有想要编译的规则。
        - **rule_name** (str) - 编译规则的名称。默认值： ``"KroneckerSeqCompiler"`` 。
        - **log_level** (int) - log信息展示级别。可以为 ``0`` 、 ``1`` 或者 ``2`` 。关于log级别的更多信息，请参考 :class:`~.algorithm.compiler.BasicCompilerRule` 。

    .. py:method:: do(dag_circuit: DAGCircuit)

        原位的将该编译规则运用到 :class:`~.algorithm.compiler.DAGCircuit` 上。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 量子线路的 DAG 图表示。
