.. py:class:: mindquantum.core.gates.FSim(theta: ParameterResolver, phi: ParameterResolver)

    FSim 门表示费米子模拟门。FSim 门的矩阵形式为：

    .. math::

        {\rm FSim}(\theta, \phi)=\begin{pmatrix}1&0&0&0\\0&\cos(\theta)&-i\sin(\theta)&0\\
            0&-i\sin(\theta)&\cos(\theta)&0\\0&0&0&e^{i\phi}\end{pmatrix}

    参数：
        - **theta** (Union[numbers.Number, dict, ParameterResolver]) - FSim 门的第一个参数。
        - **phi** (Union[numbers.Number, dict, ParameterResolver]) - FSim 门的第二个参数。
