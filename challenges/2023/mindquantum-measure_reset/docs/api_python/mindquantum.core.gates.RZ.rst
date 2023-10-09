mindquantum.core.gates.RZ
===============================

.. py:class:: mindquantum.core.gates.RZ(pr)

    围绕z轴的旋转门。更多用法，请参见 :class:`~.core.gates.RX`。

    .. math::

        {\rm RZ}=\begin{pmatrix}\exp(-i\theta/2)&0\\
                         0&\exp(i\theta/2)\end{pmatrix}

    参数：
        - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: get_cpp_obj()

        返回该门的c++对象。
