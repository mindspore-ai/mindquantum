.. py:class:: mindquantum.core.gates.ZZ(pr)

    伊辛ZZ门。更多用法，请参见 :class:`mindquantum.core.gates.RX`。

    .. math::

        {\rm ZZ_\theta}=\cos(\theta)I\otimes I-i\sin(\theta)\sigma_Z\otimes\sigma_Z

    **参数：**

    - **pr** (Union[int, float, str, dict, ParameterResolver]) - 参数化门的参数，详细解释请参见上文。

    .. py:method:: matrix(pr=None, frac=1)

        返回ZZ门的矩阵。

    .. py:method:: diff_matrix(pr=None, about_what=None, frac=1)

        返回ZZ门的导数矩阵。
