mindquantum.core.gates.RZ
=========================

.. py:class:: mindquantum.core.gates.RZ(coeff=None)

    围绕z轴的旋转门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm RZ}=\begin{pmatrix}\exp(-i\theta/2)&0\\
                         0&\exp(i\theta/2)\end{pmatrix}

    **参数：**

    - **coeff** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。默认值：None。
    