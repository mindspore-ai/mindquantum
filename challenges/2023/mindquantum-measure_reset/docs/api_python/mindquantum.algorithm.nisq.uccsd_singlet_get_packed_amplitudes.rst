mindquantum.algorithm.nisq.uccsd_singlet_get_packed_amplitudes
===============================================================

.. py:function:: mindquantum.algorithm.nisq.uccsd_singlet_get_packed_amplitudes(single_amplitudes, double_amplitudes, n_qubits, n_electrons)

    提取单态UCCSD算子的振幅系数。输出列表仅包含与单态UCCSD相关的振幅，顺序与 `uccsd_singlet_generator` 保持一致。

    参数：
        - **single_amplitudes** (numpy.ndarray) - :math:`N\times N` 维的数组，该数组存储着 :math:`t_{i,j} * (a_i^\dagger a_j - \text{H.C.})` 和对应的排序好的单激发算符的振幅。
        - **double_amplitudes** (numpy.ndarray) - :math:`N\times N\times N\times N` 数组，该数组存储着 :math:`t_{i,j,k,l} * (a_i^\dagger a_j a_k^\dagger a_l - \text{H.C.})` 和对应的排序好的双激发算符的振幅。
        - **n_qubits** (int) - 用于表示系统的自旋轨道数，这也对应于非紧凑映射中的量子比特数。
        - **n_electrons** (int) - 物理系统中电子的数量。

    返回：
        ParameterResolver，存储着所有单激发态和双激发态算符的系数。在返回的系数中，单激发态系数位于双激发态之前。
