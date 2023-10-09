mindquantum.algorithm.compiler.GateReplacer
===========================================

.. py:class:: mindquantum.algorithm.compiler.GateReplacer(ori_example_gate: BasicGate, wanted_example_circ: Circuit)

    将给定的量子门替换成给定的量子线路。

    参数：
        - **ori_example_gate** (:class:`~.core.gates.BasicGate`) - 想要替换的量子门。请注意，相同类型且拥有相同个数的作用为和控制为的量子门会被替换。
        - **wanted_example_circ** (:class:`~.core.circuit.Circuit`) - 想要的量子线路。

    .. py:method:: do(dag_circuit: DAGCircuit)

        执行门替换规则。

        参数：
            - **dag_circuit** (:class:`~.algorithm.compiler.DAGCircuit`) - 想要编译的量子线路的 DAG 图。
