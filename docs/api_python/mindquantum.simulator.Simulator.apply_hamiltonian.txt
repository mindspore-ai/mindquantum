mindquantum.simulator.Simulator.apply_hamiltonian(hamiltonian: mindquantum.core.operators.hamiltonian.Hamiltonian)

        将汉密尔顿应用到模拟器上，这个汉密尔顿可以是隐士或非隐士。

        注:
            应用哈密顿量后，量子态可能不是归一化量子态。

        参数:
            hamiltonian (Hamiltonian): 你想申请的汉密尔顿人。

        样例:
            >>> from mindquantum import Simulator
            >>> from mindquantum import Circuit, Hamiltonian
            >>> from mindquantum.core.operators import QubitOperator
            >>> import scipy.sparse as sp
            >>> sim = Simulator('projectq', 1)
            >>> sim.apply_circuit(Circuit().h(0))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
            >>> ham1 = Hamiltonian(QubitOperator('Z0'))
            >>> sim.apply_hamiltonian(ham1)
            >>> sim.get_qs()
            array([ 0.70710678+0.j, -0.70710678+0.j])

            >>> sim.reset()
            >>> ham2 = Hamiltonian(sp.csr_matrix([[1, 2], [3, 4]]))
            >>> sim.apply_hamiltonian(ham2)
            >>> sim.get_qs()
            array([1.+0.j, 3.+0.j])
        