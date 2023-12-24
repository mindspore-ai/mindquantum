
mindquantum.core.gates.AmplitudeDampingChannel
===============================================

.. py:class:: mindquantum.core.gates.AmplitudeDampingChannel(gamma: float, **kwargs)

    振幅阻尼信道。可以表示量子比特由于能量耗散导致的错误。

    振幅阻尼信道通常可表示为：

    .. math::

        \begin{gather*}
        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger
        \\
        \text{其中}\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&\sqrt{\gamma}\\
                0&0\end{bmatrix}
        \end{gather*}

    这里 :math:`\rho` 是密度矩阵形式的量子态； :math:`\gamma` 是能量损耗系数。

    参数：
        - **gamma** (int, float) - 振幅阻尼系数。

    .. py:method:: define_projectq_gate()

        定义对应的projectq门。

    .. py:method:: get_cpp_obj()

        返回底层c++对象。

    .. py:method:: matrix()

        返回该噪声信道的Kraus算符。

        返回：
            list，包含了该噪声信道的Kraus算符。
