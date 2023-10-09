mindquantum.algorithm.compiler.SequentialCompiler
=================================================

.. py:class:: mindquantum.algorithm.compiler.SequentialCompiler(compilers: typing.List[BasicCompilerRule], rule_name="SequentialCompiler", log_level=0)

    序列化编译规则。

    序列中的编译规则会顺序化执行。

    参数：
        - **compilers** (List[:class:`~.algorithm.compiler.BasicCompilerRule`]) - 所有想要编译的规则。
        - **rule_name** (str) - 该编译规则的名称。默认值： ``"SequentialCompiler"``。
        - **log_level** (int) - log信息展示级别。可以为 ``0`` 、 ``1`` 或者 ``2`` 。关于log级别的更多信息，请参考 :class:`~.algorithm.compiler.BasicCompilerRule` 。默认值： ``0`` 。

    .. py:method:: do(dag_circuit: DAGCircuit)

        原位的将该编译规则运用到 :class:`~.algorithm.compiler.DAGCircuit` 上。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 量子线路的 DAG 图表示。

    .. py:method:: set_all_log_level(log_level: int)

        设置log信息的展示级别。

        参数：
            - **log_level** (int) - log信息展示级别。可以为 ``0`` 、 ``1`` 或者 ``2`` 。关于log级别的更多信息，请参考 :class:`~.algorithm.compiler.BasicCompilerRule` 。
