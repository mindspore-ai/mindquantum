mindquantum.core.gates.RY
===============================

.. py:class:: mindquantum.core.gates.RY(pr)

    围绕y轴的旋转门。更多用法，请参见 :class:`~.core.gates.RX`。

    .. math::

        {\rm RY}=\begin{pmatrix}\cos(\theta/2)&-\sin(\theta/2)\\
                         \sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
