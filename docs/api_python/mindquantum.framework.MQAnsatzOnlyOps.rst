.. py:class:: mindquantum.framework.MQAnsatzOnlyOps(expectation_with_grad)

    MindQuantum 算子，通过参数化量子电路 (PQC) 获得对量子态的哈密顿期望。 这个PQC应该只包含一个ansatz电路。 此操作仅受 `PYNATIVE_MODE` 支持。

    **参数：**

    - **expectation_with_grad** (GradOpsWrapper) – 接收编码器数据和ansatz数据，并返回相对于参数的期望值和梯度值。

    **输入：**

    - **ans_data** (Tensor) - shape为 :math:`N` 的Tensor，用于ansatz电路，其中 :math:`N` 表示ansatz参数的数量。

    **输出：**

    - **Output** (Tensor) - hamiltonian的期望值。   