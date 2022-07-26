mindquantum.framework.MQEncoderOnlyOps
======================================

.. py:class:: mindquantum.framework.MQEncoderOnlyOps(expectation_with_grad)

    仅包含encoder线路的量子线路演化算子。通过参数化量子线路(PQC)获得对量子态的哈密顿期望。此算子只能在 `PYNATIVE_MODE` 下执行。

    参数：
        - **expectation_with_grad** (GradOpsWrapper) - 接收encoder数据和ansatz数据，并返回期望值和参数相对于期望的梯度值。

    输入：
        - **enc_data** (Tensor) - 希望编码为量子态的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示encoder数量。

    输出：
        Tensor，hamiltonian的期望值。
