Class mindquantum.simulator.Simulator(backend, n_qubits, seed=None)

    模拟量子电路的量子模拟器。

    参数:
        backend (str): 您想要的后端。支持的后端可在SUPPORTED_SIMULATOR中找到。
        n_qubits (int): 量子模拟器的数量。
        seed (int): 此模拟器的随机种子，如果无，种子将由`numpy.random.randint`生成。默认值：None。

    异常:
        TypeError: 如果`backend`不是str。
        TypeError: 如果`n_qubits`不是int。
        TypeError: 如果`种子`不是int。
        ValueError: 如果不支持`backend`。
        ValueError: 如果`n_qubits`为负数。
        ValueError: 如果`种子`小于0或大于2**23 - 1。

    样例:
        >>> from mindquantum import Simulator
        >>> from mindquantum import qft
        >>> sim = Simulator('projectq', 2)
        >>> sim.apply_circuit(qft(range(2)))
        >>> sim.get_qs()
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
    