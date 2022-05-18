.. py:class:: mindquantum.framework.MQOps(expectation_with_grad)

    MindQuantum算子，它返回在参数化量子电路（PQC）评估出的量子态上，hamiltonian期望绝对值的平方。这个PQC应该包含一个encoder电路和一个ansatz电路。此操作仅受 `PYNATIVE_MODE` 支持。

    **参数：**

    - **expectation_with_grad** (GradOpsWrapper) – 接收encoder数据和ansatz数据，并返回期望值和参数相对于期望的梯度值。

    **输入：**

    - **enc_data** (Tensor) - 您希望编码为量子状态的Tensor，其shape为 :math:`(N, M)` ，其中 :math:`N` 表示batch大小， :math:`M` 表示encoder参数的数量。    
    - **ans_data** (Tensor) - shape为 :math:`N` 的Tensor，用于ansatz电路，其中 :math:`N` 表示ansatz参数的数量。

    **输出：**

    - **Output** (Tensor) - hamiltonian的期望值。