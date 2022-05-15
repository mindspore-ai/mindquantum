mindquantum.simulator.Simulator.get_expectation(hamiltonian)

        得到对汉密尔顿人的期望。汉密尔顿人可能是非隐士人。

        .. math::

            E = \left<\psi\right|H\left|\psi\right>

        参数:
            hamiltonian (Hamiltonian): 你想得到期望的汉密尔顿人。

        返回:
            数字。数字，期望值。

        样例:
            >>> from mindquantum.core.operators import QubitOperator
            >>> from mindquantum import Circuit, Simulator
            >>> from mindquantum import Hamiltonian
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_circuit(Circuit().ry(1.2, 0))
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim.get_expectation(ham)
            (0.36235775447667357+0j)
        