mindquantum.simulator.Simulator.set_qs(quantum_state)

        设置此模拟的量子状态。

        参数:
            quantum_state (numpy.ndarray): 您想要的量子状态。

        样例:
            >>> from mindquantum import Simulator
            >>> import numpy as np
            >>> sim = Simulator('projectq', 1)
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j])
            >>> sim.set_qs(np.array([1, 1]))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
        