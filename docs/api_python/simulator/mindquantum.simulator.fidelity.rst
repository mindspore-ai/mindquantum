mindquantum.simulator.fidelity
====================================

.. py:function:: mindquantum.simulator.fidelity(rho: np.ndarray, sigma: np.ndarray)

    计算两个量子态的保真度。

    量子态保真度的定义如下所示。

    .. math::
        F(\rho, \sigma) = \left( \text{tr} \sqrt{\sqrt{\rho} \sigma \sqrt{\rho}} \right)^2

    其中 :math:`\rho` 和 :math:`\sigma` 是密度矩阵。

    如果 :math:`\rho` 和 :math:`\sigma` 都是纯态，则有 :math:`\rho=\left|\psi_\rho\right>\!\left<\psi_\rho\right|`
    和 :math:`\sigma=\left|\psi_\sigma\right>\!\left<\psi_\sigma\right|`，此时

    .. math::
        F(\rho, \sigma) = \left| \left< \psi_\rho \middle| \psi_\sigma \right> \right|^2

    此外，该接口还支持状态向量和密度矩阵混合输入。

    参数：
        - **rho** (np.ndarray) - 其中一个量子态。支持态矢量或密度矩阵。
        - **sigma** (np.ndarray) - 另一个量子态。支持态矢量或密度矩阵。

    返回：
        numbers.Number，两个量子态的保真度。
