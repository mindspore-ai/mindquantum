.. py:class:: mindquantum.core.gates.AmplitudeDampingChannel(gamma: float, **kwargs)

    用于表征量子计算中非相干噪声的信道。
    振幅衰减信道表示的是量子比特由于能量耗散导致的错误。
    振幅衰减信道通常可表示为：

    .. math::

        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger

        where\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&\sqrt{\gamma}\\
                0&0\end{bmatrix}

    这里 :math:`\rho` 是密度矩阵形式的量子态。 :math:`\gamma` 是能量损耗系数。

    **参数：**

    - **gamma** (int, float) - 振幅衰减系数。

    .. py:method:: define_projectq_gate()

        定义对应的projectq门。
