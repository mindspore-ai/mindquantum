mindquantum.algorithm.library.qft(qubits)

    量子傅里叶变换。

    注:
        有关更多信息，请参考Nielsen, M., & Chuang, I. (2010)。

    参数:
        qubits (list[int]): 要应用量子傅里叶变换的量子位。

    样例:
        >>> from mindquantum.algorithm.library import qft
        >>> print(qft([0, 1]).get_qs(ket=True))
        1/2¦00⟩
        1/2¦01⟩
        1/2¦10⟩
        1/2¦11⟩

    返回:
        电路，可以做傅里叶变换的电路。
       