.. py:class:: mindquantum.core.gates.YY(pr)

    伊辛YY门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm YY_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_y\otimes\sigma_y

    **参数：**

    - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: matrix(pr=None, frac=1):

        返回YY门的矩阵。

    .. py:method:: diff_matrix(pr=None, about_what=None, frac=1):

        返回YY门的导数矩阵。
