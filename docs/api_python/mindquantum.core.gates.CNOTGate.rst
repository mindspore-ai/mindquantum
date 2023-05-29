mindquantum.core.gates.CNOTGate
================================

.. py:class:: mindquantum.core.gates.CNOTGate

    控制X门。

    更多用法，请参见 :class:`~.core.gates.XGate` 。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。

    .. py:method:: on(obj_qubits, ctrl_qubits=None)

        定义量子门作用在哪些量子别上，并受哪些量子比特控制。

        .. note::
            在本框架中，接口中的第一个参数是量子门作用在哪些比特上，第二个参数是控制比特，即使对于控制门也是如此，例如CNOT门。

        参数：
            - **obj_qubits** (int, list[int]) - 指明量子门作用在哪些量子比特上。
            - **ctrl_qubits** (int, list[int]) - 指明量子门受哪些量子比特控制。默认值： ``None``。

        返回：
            量子门，返回一个新的量子门。
