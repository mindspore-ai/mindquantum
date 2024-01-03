mindquantum.core.gates.U3
===============================

.. py:class:: mindquantum.core.gates.U3(theta: ParameterResolver, phi: ParameterResolver, lamda: ParameterResolver)

    U3 门表示单比特的任意幺正门。U3 门的矩阵形式为：

    .. math::

        {\rm U3}(\theta, \phi, \lambda) =\begin{pmatrix}\cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
            e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)\end{pmatrix}

    它可以被分解为：

    .. math::

        {\rm U3}(\theta, \phi, \lambda) = RZ(\phi) RX(-\pi/2) RZ(\theta) RX(\pi/2) RZ(\lambda)

    参数：
        - **theta** (Union[numbers.Number, dict, ParameterResolver]) - U3 门的第一个参数。
        - **phi** (Union[numbers.Number, dict, ParameterResolver]) - U3 门的第二个参数。
        - **lamda** (Union[numbers.Number, dict, ParameterResolver]) - U3 门的第三个参数。

    .. py:method:: get_cpp_obj()

        返回量子门的c++对象。

    .. py:method:: hermitian()

        获取 U3 门的厄米共轭形式。

    .. py:method:: lamda()
        :property:

        获取 U3 门的参数 lamda。

        返回：
            ParameterResolver，参数 lamda。

    .. py:method:: matrix(pr: ParameterResolver = None, full=False)

        获取 U3 门的矩阵形式。

        参数：
            - **pr** (Union[ParameterResolver, dict]) - U3 门的参数。默认值： ``None``。
            - **full** (bool) - 是否获取完整的矩阵（受控制比特和作用比特影响）。默认值： ``False``。

    .. py:method:: phi()
        :property:

        获取 U3 门的参数 phi。

        返回：
            ParameterResolver，参数 phi。

    .. py:method:: theta()
        :property:

        获取 U3 门的参数 theta。

        返回：
            ParameterResolver，参数 theta。
