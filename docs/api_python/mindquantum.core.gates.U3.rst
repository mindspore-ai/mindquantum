.. py:class:: mindquantum.core.gates.U3(theta, phi, lamda)

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
