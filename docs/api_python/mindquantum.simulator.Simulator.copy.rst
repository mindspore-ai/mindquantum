mindquantum.simulator.Simulator.copy()

        复制此模拟器。

        返回:
            模拟器，此模拟器的副本版本。

        样例:
            >>> from mindquantum import RX, Simulator
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(RX(1).on(0))
            >>> sim.flush()
            >>> sim2 = sim.copy()
            >>> sim2.apply_gate(RX(-1).on(0))
            >>> sim2
            projectq simulator with 1 qubit (little endian).
            Current quantum state:
            1¦0⟩
           