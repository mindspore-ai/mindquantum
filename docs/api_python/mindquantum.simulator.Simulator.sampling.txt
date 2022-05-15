mindquantum.simulator.Simulator.sampling(circuit, pr=None, shots=1, seed=None)

        在电路中对测量量子位进行模拟。采样不会改变此模拟器的原点量子状态。

        参数:
            circuit (Circuit): 要进化和采样的电路。
            pr (Union[None, dict, ParameterResolver]): 如果此电路是参数化电路，则此电路的参数解析器。默认值：None。
            shots (int): 您想采样此电路的镜头数量。默认值：1。
            seed (int): 随机抽样的随机种子。如果无，则种子将是随机的int数。默认值：None。

        返回:
            测量结果，采样的测量结果。

        样例:
            >>> from mindquantum import Circuit, Measure
            >>> from mindquantum import Simulator
            >>> circ = Circuit().ry('a', 0).ry('b', 1)
            >>> circ += Measure('q0_0').on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ += Measure('q1').on(1)
            >>> sim = Simulator('projectq', circ.n_qubits)
            >>> res = sim.sampling(circ, {'a': 1.1, 'b': 2.2}, shots=100, seed=42)
            >>> res
            shots: 100
            Keys: q1 q0_1 q0_0│0.00   0.122       0.245       0.367        0.49       0.612
            ──────────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                           000│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                              │
                           011│▒▒▒▒▒▒▒▒▒
                              │
                           100│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              │
                           111│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                              │
            {'000': 18, '011': 9, '100': 49, '111': 24}
           