import numpy as np
from mindquantum import Circuit
from mindquantum.core import X, Y, Z, H, RX, RY, RZ, Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.core import UN
from mindquantum import UnivMathGate
from mindquantum import QubitOperator
from mindquantum.framework import MQLayer
from functools import reduce

#define √iSWAP gate
iSWAP_sqrt_mat = np.array([[1, 0, 0, 0],
                           [0, 1 / np.sqrt(2), 1 / np.sqrt(2) * (0. + 1.j), 0],
                           [0, 1 / np.sqrt(2) * (0. + 1.j), 1 / np.sqrt(2), 0],
                           [0, 0, 0, 1]])
iSWAP_sqrt_gate = UnivMathGate('√iSWAP', iSWAP_sqrt_mat)


#define the function of entanglement gate arrangement of quantum circuit
def entangling_layer(n_qubits, entangling_arrangement, type_entangler, k):
    entangling_layer = Circuit()
    if entangling_arrangement == 0:  #entangling_arrangement = CHAIN
        a = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [1, 2], [3, 4], [5, 6],
             [7, 8]]
        b = [[1, 0], [3, 2], [5, 4], [7, 6], [9, 8], [2, 1], [4, 3], [6, 5],
             [8, 7]]
        rand_int = np.random.randint(0, 2, 9)
        if type_entangler == 0:  # type_entangling = CNOT
            for i in range(len(a)):
                if rand_int[i] == 0:
                    entangling_layer += X.on(a[i][0], a[i][1])
                else:
                    entangling_layer += X.on(b[i][0], b[i][1])
        elif type_entangler == 1:  # type_entangling = CPHASE
            for i in range(len(a)):
                if rand_int[i] == 0:
                    entangling_layer += Z.on(a[i][0], a[i][1])
                else:
                    entangling_layer += Z.on(b[i][0], b[i][1])
        elif type_entangler == 2:  # type_entangling = √iSWAP
            for i in range(len(a)):
                if rand_int[i] == 0:
                    entangling_layer += iSWAP_sqrt_gate.on([a[i][0], a[i][1]])
                else:
                    entangling_layer += iSWAP_sqrt_gate.on([b[i][0], b[i][1]])
    elif entangling_arrangement == 1:  #entangling_arrangement = ALL
        rand_int = np.random.randint(0, 2, 45)
        n = 0
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                if type_entangler == 0:
                    if rand_int[n] == 0:
                        entangling_layer += X.on(i, j)  # type_entangling =CNOT
                    else:
                        entangling_layer += X.on(j, i)
                elif type_entangler == 1:
                    if rand_int[n] == 0:
                        entangling_layer += Z.on(i,
                                                 j)  # type_entangling = CPHASE
                    else:
                        entangling_layer += Z.on(j, i)
                elif type_entangler == 2:
                    if rand_int[n] == 0:
                        entangling_layer += iSWAP_sqrt_gate.on(
                            [i, j])  # type_entangling = √iSWAP
                    else:
                        entangling_layer += iSWAP_sqrt_gate.on([j, i])
                n += 1
    elif entangling_arrangement == 2:  #entangling_arrangement = ALT
        if k % 2 == 1:
            a = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
            b = [[1, 0], [3, 2], [5, 4], [7, 6], [9, 8]]
            rand_int = np.random.randint(0, 2, 5)
            if type_entangler == 0:  # type_entangling = CNOT
                for i in range(len(a)):
                    if rand_int[i] == 0:
                        entangling_layer += X.on(a[i][0], a[i][1])
                    else:
                        entangling_layer += X.on(b[i][0], b[i][1])
            elif type_entangler == 1:  # type_entangling = CPHASE
                for i in range(len(a)):
                    if rand_int[i] == 0:
                        entangling_layer += Z.on(a[i][0], a[i][1])
                    else:
                        entangling_layer += Z.on(b[i][0], b[i][1])
            elif type_entangler == 2:  # type_entangling = √iSWAP
                for i in range(len(a)):
                    if rand_int[i] == 0:
                        entangling_layer += iSWAP_sqrt_gate.on(
                            [a[i][0], a[i][1]])
                    else:
                        entangling_layer += iSWAP_sqrt_gate.on(
                            [b[i][0], b[i][1]])
        elif k % 2 == 0:
            a = [[4, 5], [6, 3], [7, 2], [8, 1], [9, 0]]
            b = [[5, 4], [3, 6], [2, 7], [1, 8], [0, 9]]
            rand_int = np.random.randint(0, 2, 5)
            if type_entangler == 0:
                for i in range(len(a)):
                    if rand_int[i] == 0:
                        entangling_layer += X.on(a[i][0], a[i][1])
                    else:
                        entangling_layer += X.on(b[i][0], b[i][1])
            elif type_entangler == 1:
                for i in range(len(a)):
                    if rand_int[i] == 0:
                        entangling_layer += Z.on(a[i][0], a[i][1])
                    else:
                        entangling_layer += Z.on(b[i][0], b[i][1])
            elif type_entangler == 2:
                for i in range(len(a)):
                    if rand_int[i] == 0:
                        entangling_layer += iSWAP_sqrt_gate.on(
                            [a[i][0], a[i][1]])
                    else:
                        entangling_layer += iSWAP_sqrt_gate.on(
                            [b[i][0], b[i][1]])
    return entangling_layer


#define the function of the quantum circuit
def rot_type(angle):
    rot_int = np.random.randint(0, 3)
    if rot_int == 0:
        return RX(angle)
    elif rot_int == 1:
        return RY(angle)
    elif rot_int == 2:
        return RZ(angle)


def circuit(n_qubits, depth, entangling_arrangement, type_entangler):
    circ = Circuit()
    circ += UN(RY(np.pi / 4), n_qubits)
    for j in range(depth):
        for i in range(n_qubits):
            circ += rot_type(f'theta{j*n_qubits+i}').on(i)
        circ += entangling_layer(n_qubits, entangling_arrangement,
                                 type_entangler, j)
    return circ


#define the function of partial derivative of quantum circuit
def partial_qs(circ, partial_param, params_dict, entangling_arrangement,
               type_entangler):
    n = circ.n_qubits
    param_index = circ.params_name.index(partial_param)
    if entangling_arrangement == 0:
        gate_index = (2 * n - 1) * (param_index // n) + param_index % n + n
    elif entangling_arrangement == 1:
        gate_index = (n + reduce(lambda x, y: x + y, range(0, n))) * (
            param_index // n) + param_index % n + n
    elif entangling_arrangement == 2:
        gate_index = (n + int(n / 2)) * (param_index //
                                         n) + param_index % n + n
    gate_name = circ[gate_index].name
    obj_q = circ[gate_index].obj_qubits
    if gate_name == 'RX':
        partial_circ = circ[:(gate_index +
                              1)] + X.on(obj_q) + circ[(gate_index + 1):]
    elif gate_name == 'RY':
        partial_circ = circ[:(gate_index +
                              1)] + Y.on(obj_q) + circ[(gate_index + 1):]
    elif gate_name == 'RZ':
        partial_circ = circ[:(gate_index +
                              1)] + Z.on(obj_q) + circ[(gate_index + 1):]
    sim = Simulator('mqvector', n)
    sim.apply_circuit(partial_circ, params_dict)
    partial_derivative = (-1j / 2) * sim.get_qs()
    return partial_derivative


# define the function to obtain the matrix elements
def QFI_element(circ, partial_param1, partial_param2, params_dict,
                entangling_arrangement, type_entangler):
    sim = Simulator('mqvector', circ.n_qubits)
    sim.apply_circuit(circ, params_dict)
    qs = sim.get_qs()
    partial1 = partial_qs(circ, partial_param1, params_dict,
                          entangling_arrangement, type_entangler)
    partial2 = partial_qs(circ, partial_param2, params_dict,
                          entangling_arrangement, type_entangler)
    qfi_element = np.vdot(
        partial1, partial2) - (np.vdot(partial1, qs)) * (np.vdot(qs, partial2))
    return qfi_element


# define the function that return the quantum fisher information matrix.
def QFI(circ, params, type_entangler, entangling_arrangement):
    ParamsName = circ.params_name
    matrix_qfi = np.zeros([len(ParamsName), len(ParamsName)])
    params_dict = {}
    for (name, value) in zip(ParamsName, params):
        params_dict[name] = value
    sim = Simulator('mqvector', circ.n_qubits)
    sim.apply_circuit(circ, params_dict)
    i = 0
    for x in ParamsName:
        i += 1
        j = 0
        for y in ParamsName:
            j += 1
            matrix_qfi[i - 1, j - 1] = QFI_element(circ, x, y, params_dict,
                                                   entangling_arrangement,
                                                   type_entangler).real
    return matrix_qfi


# define sampling method to obain the variance of conventional gradient.
def get_var_partial_exp(circuit, hamiltonian, attempts_number=100):
    sim = Simulator('mqvector', circuit.n_qubits)
    parameters_number = len(circuit.params_name)
    avg_variance = 0
    grad_ops = sim.get_expectation_with_grad(hamiltonian, circuit)
    if attempts_number:
        rdm_params = np.random.rand(parameters_number) * 2 * np.pi
        avg_variance += grad_ops(rdm_params)[1][0, 0,
                                                0].real**2 / attempts_number
        attempts_number -= 1
    return avg_variance


def number_of_zero_eigval(qfi_matrix, n_parameters):
    cutoff_eigvals = 10**-18  #define all eigenvalues of quantum fisher information metric as 0
    #calculate QFI_matrix's eigvals and eigvecs
    eigvals = np.linalg.eig(qfi_matrix)[0]
    #get non-zero eigenvalues
    nonzero_eigvals = eigvals[eigvals > cutoff_eigvals]
    eff_quant_dim = len(nonzero_eigvals)
    n_zero_eigval = n_parameters - eff_quant_dim
    return n_zero_eigval


# When the number of layers of ansatz is 1、3, the corresponding gradient and R of the quantum circuits combined with different types of
# entangled gates and different permutations are obtained
def R_vargrad_result(n_qubits, depth):
    R_list = np.zeros([3, 3])
    var_grad = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            entangling_arrangement = i
            type_entangler = j
            circ = circuit(n_qubits, depth, entangling_arrangement,
                           type_entangler)
            ham = Hamiltonian(QubitOperator('Z0 Z1'))
            # define angles for circuit定义电路角度
            n_parameters = depth * n_qubits
            params = np.random.rand(n_parameters) * 2 * np.pi
            # QFI matrix
            QFI_matrix = QFI(circ, params, type_entangler,
                             entangling_arrangement).real

            variance_gradient = get_var_partial_exp(circ,
                                                    ham,
                                                    attempts_number=100)

            redundancy = number_of_zero_eigval(QFI_matrix,
                                               n_parameters) / n_parameters
            # When the depth is the same, the gradient variance and R of PQC corresponding to
            # three different entanglement gates and three different Ansatz permutations are obtained
            R_list[i][
                j] = redundancy  # Rows represent permutations and columns represent entanglement gate types
            var_grad[i][j] = variance_gradient
    return R_list, var_grad
