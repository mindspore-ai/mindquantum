mindquantum.algorithm.nisq.uccsd_singlet_get_packed_amplitudes(single_amplitudes, double_amplitudes, n_qubits, n_electrons)
转换振幅以与单子UCCSD配合使用

    输出列表仅包含与单态UCCSD相关的振幅，顺序适合与函数`uccsd_单态_生成器`一起使用。

    参数:
        single_amplitudes(numpy.ndarray):：数学：`N\乘以N`数组，存储对应于：数学：`t_{i,j}*（a_i^\匕首a_j - \文本{H.C.}）`
        double_amplitudes(numpy.ndarray): :math:`N\times N\times N\times N` 数组，存储与 :math:`t_{i,j,k,l} * (a_i^\dagger a_j a_k^\dagger a_l - \text{H.C.})`
        n_qubits(int): 用于表示系统的自旋轨道数，这也对应于非紧凑映射中的量子位数。
        n_electrons(int): 物理系统中电子的数量。

    返回:
        参数解析器，列表存储单联UCCSD运算符的唯一单激励和双激励幅度。
        顺序列出了双激励之前的唯一单激励。

    样例:
        >>> import numpy as np
        >>> from mindquantum.algorithm.nisq.chem import uccsd_singlet_get_packed_amplitudes
        >>> n_qubits, n_electrons = 4, 2
        >>> np.random.seed(42)
        >>> ccsd_single_amps = np.random.random((4, 4))
        >>> ccsd_double_amps = np.random.random((4, 4, 4, 4))
        >>> uccsd_singlet_get_packed_amplitudes(ccsd_single_amps, ccsd_double_amps,
        ...                                     n_qubits, n_electrons)
        {'s_0': 0.6011150117432088, 'd1_0': 0.7616196153287176}
    