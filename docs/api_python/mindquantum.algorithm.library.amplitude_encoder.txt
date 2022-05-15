mindquantum.algorithm.library.amplitude_encoder(x, n_qubits)

    用于幅度编码的量子电路。

    注:
        经典数据的长度应该是2的幂，否则将用0填充，向量应标准化。

    参数:
        x (list[float] or numpy.array(list[float]): 要编码的数据向量，应标准化。
        n_qubits (int): 编码器电路的量子位数。

    样例:
        >>> from mindquantum.algorithm.library import amplitude_encoder
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('projectq', 8)
        >>> encoder, parameterResolver = amplitude_encoder([0.5, -0.5, 0.5, 0.5], 8)
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        1/2¦00000000⟩
        -1/2¦00000001⟩
        1/2¦00000010⟩
        1/2¦00000011⟩
        >>> sim.reset()
        >>> encoder, parameterResolver = amplitude_encoder([0, 0, 0.5, 0.5, -0.5, 0.5], 8)
        >>> sim.apply_circuit(encoder, parameterResolver)
        >>> print(sim.get_qs(True))
        1/2¦00000010⟩
        1/2¦00000011⟩
        -1/2¦00000100⟩
        1/2¦00000101⟩
       