.. py:class:: mindquantum.core.gates.PhaseDampingChannel(gamma: float, **kwargs)

    用于表征量子计算中非相干噪声的信道。
    相位衰减信道表示的是量子比特在不跟外界产生能量交换时量子信息的损失。
    相位衰减信道通常可表示为：

    .. math::

        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger

        where\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&0\\
                0&\sqrt{\gamma}\end{bmatrix}

    这里 :math:`\rho` 是密度矩阵形式的量子态； :math:`\gamma` 是信息损失系数。

    **参数：**

    - **gamma** (int, float) - 信息损失系数。

    .. py:method:: define_projectq_gate()

        定义对应的projectq门。

    .. py:method:: get_cpp_obj()

        获取底层c++对象。
