mindquantum.core.gates.BarrierGate
===================================

.. py:class:: mindquantum.core.gates.BarrierGate(show=True)

    栅栏门会将两个量子门分开在不同的层级上。

    参数：
        - **show** (bool) - 是否展示栅栏门。默认值： ``True``.

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: on(obj_qubits, ctrl_qubits=None)

        定义该门作用在哪些量子比特上。受控位必须为 ``None``，应为栅栏门不能被其他比特控制。

        参数：
            - **obj_qubits** (int, list[int]) - 指明量子门作用在哪些量子比特上，可以是单个比特，也可以是一连串的连续比特。
            - **ctrl_qubits** (int, list[int]) - 指明量子门受哪些量子比特控制。默认值： ``None``。

        返回：
            量子门，返回一个新的量子门。
