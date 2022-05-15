mindquantum.simulator.Simulator.get_qs(ket=False)

        获取此模拟器的当前量子状态。

        参数:
            ket (bool): 是否以ket格式返回量子状态。默认值：False。

        返回:
            numpy.nd数组，当前量子状态。

        样例:
            >>> from mindquantum import qft, Simulator
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.get_qs()
            array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
        