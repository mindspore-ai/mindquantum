from mindquantum.core import Hamiltonian, RX, RY, RZ, X
from scipy.optimize import minimize
from mindquantum.simulator.simulator import Simulator
from q_ham import MolInfoProduce
from copy import deepcopy
import numpy as np
import math
import itertools
from mindquantum import TimeEvolution, Circuit
from pool import singlet_SD, pauli_pool, rename_pauli_string
from openfermion.utils import count_qubits

PAULI_DICT = {1: 'X', 2: 'Y', 3: 'Z'}


def show_memory(unit='MB', threshold=1):
    '''查看变量占用内存情况

    :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
    :param threshold: 仅显示内存数值大于等于threshold的变量
    '''
    from sys import getsizeof
    scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    for i in list(globals().keys()):
        memory = eval("getsizeof({})".format(i)) // scale
        if memory >= threshold:
            print(i, memory)


def iter_qsubset(length, indices):
    for qs in itertools.combinations(range(0, len(indices)), length):
        qsubset = []
        for i in qs:
            qsubset.append(indices[i])
        yield qsubset


def iter_all_qsubset_pauli_by_length(length, indices):
    for qsubset in iter_qsubset(length, indices):
        for pauli in itertools.product(range(1, 4), repeat=length):
            yield qsubset, pauli


def generate_pauli_pool(n_qubit, max_length):
    for length in [i + 2 for i in range(max_length - 1)]:
        for qsubset, pauliword in iter_all_qsubset_pauli_by_length(
                length, range(n_qubit)):
            paulistring = [(qsubset[i], PAULI_DICT[pauliword[i]])
                           for i in range(len(qsubset))]
            yield paulistring


class IterQCC(MolInfoProduce):
    def __init__(self,
                 mole_name,
                 geometry,
                 basis,
                 charge,
                 multiplicity,
                 pauli_pool,
                 fermion_transform="jordan_wigner"):
        super(IterQCC, self).__init__(geometry, basis, charge, multiplicity,
                                      fermion_transform)
        self.molecule_name = mole_name
        self.pauli_pool = pauli_pool
        self.pauli_strings_seq = []
        self.pqc = None
        self.circuit = Circuit()
        self.paras = []

        self.step_gradients = []
        self.step_energies = []
        self.maxiter = 12
        self.iteration = None

    def gradients(self, pauli_string):
        gradients_circuit = []
        gradients_pqc = []
        gradients_circuit = self.circuit + TimeEvolution(pauli_string).circuit

        # print(gradients_circuit.params_name)
        # print(gradients_circuit)
        gradients_pqc = Simulator(
            'projectq', self.n_qubits).get_expectation_with_grad(
                self.sparsed_qubit_hamiltonian, gradients_circuit)
        plus = deepcopy(self.paras)
        minus = deepcopy(self.paras)
        plus.append(np.pi / 4)
        minus.append(-np.pi / 4)

        plus_data = np.array(plus)
        e_plus, grad_plus = gradients_pqc(plus_data)
        minus_data = np.array(minus)
        e_minus, grad_minus = gradients_pqc(minus_data)
        # print(-abs(e_plus[0, 0] - e_minus[0, 0]))
        return -abs(e_plus[0, 0] - e_minus[0, 0])

    def select_pauli_string(self):
        step_result = []

        for pauli_string in self.pauli_pool:
            pauli_string = rename_pauli_string(pauli_string, self.iteration)
            gra = self.gradients(pauli_string)
            #print(pauli_string, gra)
            if abs(gra) > 1e-6:
                step_result.append([gra, pauli_string])

        step_result = self.step_result_sorting(step_result)
        #print(step_result)
        string = step_result[0][1]
        gradient = step_result[0][0]
        self.pauli_strings_seq.append(string)
        self.step_gradients.append(gradient)

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
        print(self.step_energies)
        self.paras = result.x.tolist()

    def energy(self, paras):
        ansatz_data = np.array(paras)
        e, grad = self.pqc(ansatz_data)
        return np.real(e[0, 0]), np.real(grad[0, 0])

    def generate_qmf(self):
        # TODO
        circ = Circuit()
        for qubit in range(self.n_qubits):
            circ += RY(f'{qubit}').on(qubit)
        self.pqc = Simulator('projectq',
                             self.n_qubits).get_expectation_with_grad(
                                 self.sparsed_qubit_hamiltonian, circ)
        paras = [math.pi / 4 for terms in circ.params_name]
        result = minimize(self.energy,
                          paras,
                          method='BFGS',
                          jac=True,
                          tol=1e-6,
                          options={'disp': True})
        paras = result.x.tolist()
        encoder_circuit = Circuit()
        for qubit in range(self.n_qubits):
            encoder_circuit += RX(paras[qubit]).on(qubit)
        return encoder_circuit

    def process(self):
        #encoder_circuit = self.generate_qmf()
        self.circuit = Circuit([X.on(i) for i in range(self.n_electrons)])
        for iteration in range(self.maxiter):

            self.iteration = iteration
            self.select_pauli_string()
            self.circuit += TimeEvolution(self.pauli_strings_seq[-1]).circuit
            self.pqc = Simulator(
                'projectq', self.n_qubits).get_expectation_with_grad(
                    self.sparsed_qubit_hamiltonian, self.circuit)
            self.paras_optimize()
            if abs(self.step_energies[-1] - self.fci_energy) < 0.0016:
                print('Reach chamical accuracy')
                break

            if iteration == self.maxiter - 1:
                print('Not reach chamical accuracy but with maximum iteration')
                break