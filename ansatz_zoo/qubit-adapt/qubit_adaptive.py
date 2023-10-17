import math
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from mindquantum.simulator.simulator import Simulator
from q_ham import MolInfoProduce
from openfermion.utils import count_qubits
from mindquantum import QubitOperator
from mindquantum import TimeEvolution, Circuit
from mindquantum.core import Hamiltonian, RX, RY, RZ, X
from pool import singlet_SD, pauli_pool, rename_pauli_string


class QubitAdaptive(MolInfoProduce):
    def __init__(self,
                 mole_name,
                 geometry,
                 basis,
                 charge,
                 multiplicity,
                 pauli_pool,
                 fermion_transform="jordan_wigner"):
        super(QubitAdaptive, self).__init__(geometry, basis, charge,
                                            multiplicity, fermion_transform)
        self.molecule_name = mole_name
        self.pauli_pool = pauli_pool
        self.pauli_strings_seq = []
        self.pqc = None
        self.gradients_pqc = None

        self.init_circuit = Circuit([X.on(i) for i in range(self.n_electrons)])

        self.circuit = deepcopy(self.init_circuit)
        self.paras = []

        self.step_gradients = []
        self.step_energies = []
        self.maxiter = 20
        self.iteration = None

    def gradients(self, pauli_string):

        gradients_circuit = self.circuit + TimeEvolution(pauli_string).circuit

        # print(gradients_circuit.params_name)
        # print(gradients_circuit)
        self.gradients_pqc = Simulator(
            'mqvector', self.n_qubits).get_expectation_with_grad(
                self.sparsed_qubit_hamiltonian, gradients_circuit)
        plus = deepcopy(self.paras)
        minus = deepcopy(self.paras)
        plus.append(np.pi / 4)
        minus.append(-np.pi / 4)

        plus_data = np.array(plus)
        e_plus, grad_plus = self.gradients_pqc(plus_data)
        minus_data = np.array(minus)
        e_minus, grad_minus = self.gradients_pqc(minus_data)
        # print(-abs(e_plus[0, 0] - e_minus[0, 0]))
        return -abs(e_plus[0, 0] - e_minus[0, 0])

    def select_pauli_string(self):
        step_result = []

        for pauli_string in self.pauli_pool:
            pauli_string = rename_pauli_string(pauli_string, self.iteration)
            gra = self.gradients(pauli_string)
            if abs(gra) > 1e-6:
                step_result.append([gra, pauli_string])

        step_result = self.step_result_sorting(step_result)
        # print(step_result)
        # print(len(step_result[0]))

        self.pauli_strings_seq.append(step_result[0][1])
        self.step_gradients.append(step_result[0][0])

    def step_result_sorting(self, step_result):
        flag = True
        sorted_result = []
        # find pauli strings with the same gradients.
        for res in step_result:
            # print(sorted_result, res)

            if len(sorted_result) == 0:
                sorted_result.append(res)
            else:
                for item in sorted_result:
                    if math.isclose(item[0], res[0], abs_tol=1e-9):
                        item.append(res[1])
                        flag = False
                if flag:
                    sorted_result.append(res)
            flag = True

        # sorting pauli strings by the value of gradients
        sorted_result = sorted(sorted_result, key=lambda x: x[0])

        return sorted_result

    def paras_optimize(self):
        self.paras.append(0.0)
        result = minimize(self.energy,
                          self.paras,
                          method='BFGS',
                          jac=True,
                          tol=1e-6)
        self.step_energies.append(float(result.fun))
        self.paras = result.x.tolist()

    def energy(self, paras):

        ansatz_data = np.array(paras)
        e, grad = self.pqc(ansatz_data)
        return np.real(e[0, 0]), np.real(grad[0, 0])

    def process(self):

        for iteration in range(self.maxiter):

            self.iteration = iteration
            self.select_pauli_string()
            self.circuit += TimeEvolution(self.pauli_strings_seq[-1]).circuit
            self.pqc = Simulator('mqvector',
                                 self.n_qubits).get_expectation_with_grad(
                                     self.sparsed_qubit_hamiltonian,
                                     self.circuit)
            self.paras_optimize()

            if abs(self.step_energies[-1] - self.fci_energy) < 0.0016:
                print('Reach chamical accuracy')
                break

            if iteration == self.maxiter - 1:
                print('Not reach chamical accuracy but with maximum iteration')
                break