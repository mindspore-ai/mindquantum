mindquantum.algorithm.library.general_ghz_state(qubits)

    基于零状态准备一般GHZ状态的电路。

    参数:
        qubits (list[int]): 要应用通用GHZ状态的量子位。

    样例:
        >>> from mindquantum.algorithm.library import general_ghz_state
        >>> print(general_ghz_state(range(3)).get_qs(ket=True))
        √2/2¦000⟩
        √2/2¦111⟩
        >>> print(general_ghz_state([1, 2]).get_qs(ket=True))
        √2/2¦000⟩
        √2/2¦110⟩

    返回:
        电路，可以准备ghz状态的电路。
       