mindquantum.core.gates.BasicGate
=================================

.. py:class:: mindquantum.core.gates.BasicGate(name, n_qubits, obj_qubits=None, ctrl_qubits=None)

    BasicGate是所有门的基类。

    参数：
        - **name** (str) - 此门的名称。
        - **n_qubits** (int) - 这个门有多少个量子比特。
        - **obj_qubits** (int, list[int]) - 具体门作用在哪个量子比特上。
        - **ctrl_qubits** (int, list[int]) - 指定控制量子比特。默认值： ``None``。

    .. py:method:: acted
        :property:

        检查此门是否已经作用在量子比特上。

    .. py:method:: define_projectq_gate()

        定义对应的 `projectq` 中的量子门。

    .. py:method:: get_cpp_obj()
        :abstractmethod:

        获取底层c++对象。

    .. py:method:: hermitian()

        返回该量子门的厄米共轭形式。

    .. py:method:: matrix(*args)

        门的矩阵。

    .. py:method:: no_grad()

        设置该量子门在梯度计算相关算法中不计算梯度。

    .. py:method:: on(obj_qubits, ctrl_qubits=None)

        定义门作用于哪个量子比特和控制量子比特。

        .. note::
            在此框架中，首先指定门作用的量子比特，即使对于控制门，例如CNOT，第二个参数是控制量子比特。

        参数：
            - **obj_qubits** (int, list[int]) - 指定门作用在哪个量子比特上。
            - **ctrl_qubits** (int, list[int]) - 指定控制量子比特。默认值： ``None``。

        返回：
            返回一个新的门。

    .. py:method:: parameterized
        :property:

        检查此门是否为参数化门。

    .. py:method:: requires_grad()

        设置该量子门在梯度计算相关算法中要计算梯度。在默认情况下，参数化量子门在构造时就是需要计算梯度。
