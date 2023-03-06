mindquantum.core.gates.RX
===============================

.. py:class:: mindquantum.core.gates.RX(pr)

    围绕x轴的旋转门。

    .. math::

        {\rm RX}=\begin{pmatrix}\cos(\theta/2)&-i\sin(\theta/2)\\
                       -i\sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    该旋转门可以用三种不同的方式初始化。

    1. 如果用单个数字初始化它，那么它将是一个具有一定旋转角度的非参数化门。

    2. 如果使用单个str初始化它，那么它将是只有一个参数的参数化门，默认系数为1。

    3. 如果使用dict初始化它，例如 `{'a'：1，'b'：2}` ，则此门可以包含多个具有特定系数的参数。在这种情况下，它可以表示为：


    .. math::

        RX(a+2b)

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
