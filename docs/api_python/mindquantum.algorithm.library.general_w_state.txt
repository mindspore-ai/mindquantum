mindquantum.algorithm.library.general_w_state(qubits)

    W州将军。

    注:
        请参考https://quantumcomputing.stackexchange.com/questions/4350/general-construction-of-w-n-state.

    参数:
        qubits (list[int]): 要应用一般W状态的量子位。

    样例:
        >>> from mindquantum.algorithm.library import general_w_state
        >>> print(general_w_state(range(3)).get_qs(ket=True))
        0.5773502691896257¦001⟩
        0.5773502691896258¦010⟩
        0.5773502691896257¦100⟩

    返回:
        电路，可以准备w状态的电路。
       