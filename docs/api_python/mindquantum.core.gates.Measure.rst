.. py:class:: mindquantum.core.gates.Measure(name='')

    测量量子位的测量门。

    **参数：**

    - **name** (str) - 此测量门的键。在量子电路中，不同测量门的键应该是唯一的。默认值：``''``。

    .. py:method:: hermitian()

        厄米特门的测量，返回其自身。
