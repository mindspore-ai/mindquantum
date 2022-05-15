mindquantum.algorithm.library.bitphaseflip_operator(phase_inversion_index, n_qubits)

    此运算符生成一个可以翻转任何计算基的符号的电路。

    参数:
        phase_inversion_index (list[int]): 计算基的索引希望翻转相位。
        n_qubits (int): 量子位的总数。

    样例:
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum import UN, H, Z
        >>> from mindquantum.algorithm.library import bitphaseflip_operator
        >>> circuit = Circuit()
        >>> circuit += UN(H, 3)
        >>> circuit += bitphaseflip_operator([1, 3], 3)
        >>> print(circuit.get_qs(ket=True))
        √2/4¦000⟩
        -√2/4¦001⟩
        √2/4¦010⟩
        -√2/4¦011⟩
        √2/4¦100⟩
        √2/4¦101⟩
        √2/4¦110⟩
        √2/4¦111⟩

    返回:
        电路，位相位翻转电路。
       