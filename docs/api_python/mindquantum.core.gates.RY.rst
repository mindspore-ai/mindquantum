.. py:class:: mindquantum.core.gates.RY(coeff=None)

    围绕y轴的旋转门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm RY}=\begin{pmatrix}\cos(\theta/2)&-\sin(\theta/2)\\
                         \sin(\theta/2)&\cos(\theta/2)\end{pmatrix}

    **参数：**
    - **coeff** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。默认值：None。
    