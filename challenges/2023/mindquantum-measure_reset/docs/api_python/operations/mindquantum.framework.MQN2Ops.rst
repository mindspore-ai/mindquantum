mindquantum.framework.MQN2Ops
=============================

.. py:class:: mindquantum.framework.MQN2Ops(expectation_with_grad)

    包含encoder和ansatz线路的量子线路演化算子，算子返回在参数化量子线路（PQC）演化出的量子态上哈密顿量期望绝对值的平方。此算子只能在 `PYNATIVE_MODE` 下执行。

    .. math::

        O = \left|\left<\varphi\right| U^\dagger_l H U_r\left|\psi\right>\right|^2

    参数：
        - **expectation_with_grad** (GradOpsWrapper) - 接收encoder数据和ansatz数据，并返回期望值和参数相对于期望的梯度值。

    输入：
        - **enc_data** (Tensor) - 希望编码为量子态的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示encoder参数的数量。
        - **ans_data** (Tensor) - shape为 :math:`N` 的Tensor，用于ansatz电路，其中 :math:`N` 表示ansatz参数的数量。

    输出：
        Tensor，hamiltonian期望绝对值的平方。
