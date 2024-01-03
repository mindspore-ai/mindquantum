mindquantum.algorithm.compiler.BasicCompilerRule
================================================

.. py:class:: mindquantum.algorithm.compiler.BasicCompilerRule(rule_name="BasicCompilerRule", log_level=0)

    编译规则的基类。

    编译规则会处理量子线路的DAG图 :class:`~.algorithm.compiler.DAGCircuit` ，并且根据编译规则中的 `do` 方法来进行编译。当继承子类编译规则时，你必须实现 `do` 方法。请确保 `do` 方法会返回一个 `bool` 值，该值表示编译规则是否成功执行。

    参数：
        - **rule_name** (str) - 该编译规则的名称。
        - **log_level** (int) - 展示log信息的级别。如果为 ``0``，log信息不会被展示。如果为 ``1``，log信息展示较为简洁。如果为 ``2``，log信息展示较为丰富。默认值： ``0``。

    .. py:method:: do(dag_circuit: DAGCircuit)
        :abstractmethod:

        原位的将该编译规则运用到 :class:`~.algorithm.compiler.DAGCircuit` 上。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 量子线路的 DAG 图。

    .. py:method:: set_log_level(log_level: int)

        设置log信息的展示级别。

        参数：
            - **log_level** (int) - log信息展示级别。可以为 ``0`` 、 ``1`` 或者 ``2`` 。关于log级别的更多信息，请参考 :class:`~.algorithm.compiler.BasicCompilerRule` 。
