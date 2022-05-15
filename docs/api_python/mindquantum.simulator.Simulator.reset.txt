mindquantum.simulator.Simulator.reset()

        将模拟器重置为零状态。

        样例:
            >>> from mindquantum import Simulator
            >>> from mindquantum import qft
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.reset()
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        