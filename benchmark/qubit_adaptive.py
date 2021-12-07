from mindspore import Tensor
from mindquantum.gate import Hamiltonian, RX, RY, RZ, X
from mindquantum.nn import generate_pqc_operator
from scipy.optimize import minimize
from q_ham import MolInfoProduce
from copy import deepcopy
import numpy as np
import math
from mindquantum.ops import QubitOperator
from mindquantum.circuit import TimeEvolution, Circuit
from pool import singlet_SD, pauli_pool, rename_pauli_string
from openfermion.utils import count_qubits


class QubitAdaptive(MolInfoProduce):

    def __init__(self, mole_name, geometry, basis, charge, multiplicity, 
                pauli_pool, fermion_transform="jordan_wigner"):
        super(QubitAdaptive, self).__init__(geometry, basis, 
                                                charge, multiplicity, fermion_transform)
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

        encoder_circuit = RX("null").on(self.n_qubits-1)

        gradients_circuit = self.circuit + TimeEvolution(pauli_string).circuit
        
        # print(gradients_circuit.para_name)
        # print(gradients_circuit)
        self.gradients_pqc = generate_pqc_operator(["null"], gradients_circuit.para_name, \
                                    encoder_circuit + gradients_circuit, \
                                    Hamiltonian(self.qubit_hamiltonian))
        plus = deepcopy(self.paras)
        minus = deepcopy(self.paras)
        plus.append(np.pi/4)
        minus.append(-np.pi/4)
        encoder_data = Tensor(np.array([[0]]).astype(np.float32))
        plus_data = Tensor(np.array(plus).astype(np.float32))
        e_plus, _, grad_plus = self.gradients_pqc(encoder_data, plus_data)
        minus_data = Tensor(np.array(minus).astype(np.float32))
        e_minus, _, grad_minus = self.gradients_pqc(encoder_data, minus_data)
        # print(-abs(e_plus.asnumpy()[0, 0] - e_minus.asnumpy()[0, 0]))
        return -abs(e_plus.asnumpy()[0, 0] - e_minus.asnumpy()[0, 0])
    
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
                    if math.isclose(item[0], res[0],abs_tol=1e-9):
                        item.append(res[1])
                        flag = False
                if flag: 
                    sorted_result.append(res)
            flag = True        
            

        # sorting pauli strings by the value of gradients
        sorted_result = sorted(sorted_result, key=lambda x:x[0])

        return sorted_result

    def paras_optimize(self):
        self.paras.append(0.0)
        result = minimize(self.energy, self.paras, method='BFGS',jac=True, tol=1e-6)
        self.step_energies.append(float(result.fun))
        self.paras = result.x.tolist()

    def energy(self, paras):
        encoder_data = Tensor(np.array([[0]]).astype(np.float32))
        ansatz_data = Tensor(np.array(paras).astype(np.float32))

        e, _, grad = self.pqc(encoder_data, ansatz_data)
        return e.asnumpy()[0, 0], grad.asnumpy()[0, 0]

    def process(self):

        for iteration in range(self.maxiter):

            self.iteration = iteration
            self.select_pauli_string()
            self.circuit += TimeEvolution(self.pauli_strings_seq[-1]).circuit

            encoder_circuit = RX("null").on(self.n_qubits-1)

            self.pqc = generate_pqc_operator(["null"], self.circuit.para_name, \
                                            encoder_circuit + self.circuit, Hamiltonian(self.qubit_hamiltonian))
            self.paras_optimize()

            if abs(self.step_energies[-1] - self.fci_energy) < 0.0016:
                print('Reach chamical accuracy')
                break

            if iteration == self.maxiter - 1:
                print('Not reach chamical accuracy but with maximum iteration')
                break