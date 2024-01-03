mindquantum.core.gates.Rn
===============================

.. py:class:: mindquantum.core.gates.Rn(alpha: ParameterResolver, beta: ParameterResolver, gamma: ParameterResolver)

    Rn 门表示绕布洛赫球中给定轴旋转的量子门。Rn 门的矩阵形式为：

    .. math::

        \begin{aligned}
            {\rm Rn}(\alpha, \beta, \gamma)
                &= e^{-i(\alpha \sigma_x + \beta \sigma_y + \gamma \sigma_z)/2}\\
                &= \cos(f/2)I-i\sin(f/2)(\alpha \sigma_x + \beta \sigma_y + \gamma \sigma_z)/f\\
                &\text{where } f=\sqrt{\alpha^2 + \beta^2 + \gamma^2}
        \end{aligned}

    参数：
        - **alpha** (Union[numbers.Number, dict, ParameterResolver]) - Rn 门的第一个参数。
        - **beta** (Union[numbers.Number, dict, ParameterResolver]) - Rn 门的第二个参数。
        - **gamma** (Union[numbers.Number, dict, ParameterResolver]) - Rn 门的第三个参数。

    .. py:method:: alpha()
        :property:

        获取 Rn 门的参数 alpha。

        返回：
            ParameterResolver，参数 alpha。

    .. py:method:: beta()
        :property:

        获取 Rn 门的参数 beta。

        返回：
            ParameterResolver，参数 beta。

    .. py:method:: gamma()
        :property:

        获取 Rn 门的参数 gamma。

        返回：
            ParameterResolver，参数 gamma。

    .. py:method:: get_cpp_obj()

        返回量子门的c++对象。

    .. py:method:: hermitian()

        获取 Rn 门的厄米共轭形式。

    .. py:method:: matrix(pr: ParameterResolver = None, full=False)

        获取 Rn 门的矩阵形式。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - Rn 门的参数。默认值： ``None``。
            - **full** (bool) - 是否获取完整的矩阵（受控制比特和作用比特影响）。默认值： ``False``。
