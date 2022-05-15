mindquantum.simulator.Simulator.flush()

        用于projectq模拟器的冲洗门。项目q模拟器将缓存几个门，并将这些门融合到一个更大的门中，并作用于量子状态。
        刷新命令将要求模拟器融合当前存储的栅极并对量子状态采取行动。

        样例:
            >>> from mindquantum import Simulator
            >>> from mindquantum import H
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_gate(H.on(0))
            >>> sim.flush()
        