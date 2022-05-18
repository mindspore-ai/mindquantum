mindquantum.simulator.Simulator.apply_circuit(circuit, pr=None)

        在此模拟器上应用电路。

        参数:
            circuit (Circuit): 要应用在此模拟器上的量子电路。
            pr (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): 此电路的参数解析器。如果电路未参数化，则此参数应为无。默认值：None。

        返回:
            测量结果或无，如果电路具有测量门，则返回测量结果，否则返回无。

        样例:
            >>> import numpy as np
            >>> from mindquantum import Circuit, H
            >>> from mindquantum import Simulator
            >>> sim = Simulator('projectq', 2)
            >>> sim.apply_circuit(Circuit().un(H, 2))
            >>> sim.apply_circuit(Circuit().ry('a', 0).ry('b', 1), np.array([1.1, 2.2]))
            >>> sim
            projectq simulator with 2 qubits  (little endian).
            Current quantum state:
            -0.0721702531972066¦00⟩
            -0.30090405886869676¦01⟩
            0.22178317006196263¦10⟩
            0.9246947752567126¦11⟩
            >>> sim.apply_circuit(Circuit().measure(0).measure(1))
            shots: 1
            Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
            ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                     11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                       │
            {'11': 1}
