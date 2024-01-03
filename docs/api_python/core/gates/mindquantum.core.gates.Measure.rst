mindquantum.core.gates.Measure
===============================

.. py:class:: mindquantum.core.gates.Measure(name='', reset_to=None)

    测量量子比特的测量门。

    参数：
        - **name** (str) - 此测量门的键。在量子线路中，不同测量门的键应该是唯一的。默认值： ``''``。
        - **reset_to** (Union[int, None]) - 将量子比特重置为0态或者1态。如果是 ``None``，则不重置量子比特。默认值： ``None``。

    .. py:method:: get_cpp_obj()

        获取测量门的底层c++对象。

    .. py:method:: hermitian()

        测量门的厄米形式，返回其自身。

    .. py:method:: on(obj_qubits, ctrl_qubits=None)

        定义测量门作用在什么量子比特上。

        参数：
            - **obj_qubits** (Union[int, list[int]]) - 对哪个比特进行测量。
            - **ctrl_qubits** (Union[int, list[int]]) - 测量门不允许设置控制位。

        返回：
            Measure，以及定义好作用在哪个比特上的测量门。
