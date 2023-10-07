mindquantum.core.gates.Power
===============================

.. py:class:: mindquantum.core.gates.Power(gate, exponent=0.5)

    作用在非参数化门上的指数运算符。

    参数：
        - **gates** (:class:`~.core.gates.NoneParameterGate`) - 要作用指数运算符的基本门。
        - **exponent** (int, float) - 指数。默认值： ``0.5``。

    .. py:method:: get_cpp_obj()

        返回量子门的c++对象。
