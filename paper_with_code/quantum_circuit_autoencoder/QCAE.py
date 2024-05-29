import time

import numpy as np
from itertools import combinations

from mindquantum.simulator import Simulator, fidelity
from mindquantum.core.circuit import Circuit
from mindquantum.utils import random_circuit
from mindquantum.core.gates import I, X, Y, Z, H, RX, RY, RZ, CNOT, Measure, SWAP
from mindquantum.core.operators import QubitOperator, Hamiltonian

from scipy.optimize import minimize
from optimparallel import minimize_parallel

from QCAE_state import get_choi_state, get_ME_state


class QCAE:
    def __init__(self,
                 num_qubits: int = None,
                 num_latent: int = None,
                 num_trash: int = None) -> None:
        '''
        initialize qcae model with several qubits numbers;
        :param num_qubits: the number of qubits of the target circuit(channel) needed to be compressed;
        :param num_latent: the number of qubits of the latent circuit(channel) after encoding;
        :param num_trash: the number of qubits of the ``trash" circuit(channel) after encoding;
        num_qubits = num_latent + num_trash;
        '''
        self.num_latent = num_latent
        self.num_trash = num_trash
        if num_qubits == None:
            self.num_qubits = self.num_latent + self.num_trash
        else:
            self.num_qubits = num_qubits
        try:
            if not (self.num_qubits == self.num_latent + self.num_trash):
                raise ValueError("num_qubits is the summation of num_latent and num_trash!")
        except ValueError as e:
            print("raise exception:", repr(e))
        self.pqc_ancila_num = 0

    def initial_state(self):
        '''
        prepare the initial state circuit
        :return: initial_state_circuit : numpy.array
        '''
        a = np.ones(2 ** self.num_latent)
        rho_mixed = np.diag(a)
        rho_entangled = self.get_ME_state(self.num_trash)

        self.ME_mat = rho_entangled
        rho_initial = np.kron(rho_entangled, rho_mixed)
        return rho_initial

    def get_ME_state(self, num_qubits):
        qc = Circuit()
        for i in range(num_qubits):
            qc += H.on(i)
        for i in range(num_qubits):
            qc += CNOT.on(i + num_qubits, i)
        sim = Simulator('mqmatrix', qc.n_qubits)
        sim.apply_circuit(qc)
        ME_state = sim.get_qs()
        return ME_state

    def ansatz(self, num_qubits, param_prefix, reps=5):
        '''
        construct the ansatz circuit for the encoding process;
        :return: ansatz: Circuit
        '''
        qc = Circuit()

        for rep in range(reps):
            for i in range(num_qubits):
                qc += RY(param_prefix + str(i + num_qubits * rep)).on(i)
                qc += RZ(param_prefix + str(i + num_qubits * (rep + 1))).on(i)

            pair_list = list(combinations([i for i in range(num_qubits)], 2))
            for pair in pair_list:
                qc += X.on(pair[1], pair[0])

            for i in range(num_qubits):
                qc += RY(param_prefix + str(i + (rep + 2) * num_qubits)).on(i)
                qc += RZ(param_prefix + str(i + (rep + 3) * num_qubits)).on(i)
        return qc

    def construct_circuit(self, target_op: Circuit):
        '''
        construct the circuit in training process;
        the construction is:
            initial_state_circuit + encoding_ansatz_left + target_op + encoding_ansatz_right
        :param: target_op: the circuit(channel needed to process)
        :return: parameterized quantum circuit need to be training
        '''
        encoder_left = self.ansatz(num_qubits=target_op.n_qubits, param_prefix='theta')
        encoder_right = self.ansatz(num_qubits=target_op.n_qubits, param_prefix='beta')

        execute_circuit = Circuit()
        execute_circuit += encoder_left
        execute_circuit += target_op
        execute_circuit += encoder_right

        return execute_circuit

    def ham(self):
        hams = []
        qubit_op = QubitOperator('', 0)
        for i in range(self.num_latent, self.num_qubits):
            m, n = i, i + self.num_trash

            h_1 = Hamiltonian(QubitOperator(f'Z{m}' + ' ' + f'Z{n}', 1 / 4))
            h_2 = Hamiltonian(QubitOperator(f'X{m}' + ' ' + f'X{n}', 1 / 4))
            h_3 = Hamiltonian(QubitOperator(f'Y{m}' + ' ' + f'Y{n}', -1 / 4))
            hams.append(h_1)
            hams.append(h_2)
            hams.append(h_3)

        self.hams = hams
        return qubit_op

    def cost_func(self, params):
        expec_sum = 0
        for target_op in self.target_op_list:
            circuit = self.construct_circuit(target_op=target_op)
            for i in range(self.num_qubits, self.num_qubits + self.num_trash):
                circuit += I.on(i)

            sim = Simulator('mqmatrix', circuit.n_qubits)

            rho_initial = self.initial_state()

            sim.set_qs(rho_initial)

            pairs = zip(circuit.params_name, params)
            pr = {k: v for k, v in pairs}
            sim.apply_circuit(circuit, pr=pr)

            expectation = 0
            offset = int(len(self.hams) / 3) / 4
            for h in self.hams:
                expectation += sim.get_expectation(h).real
            expec = offset + expectation
            expec_sum += expec
        return 1 - expec_sum / len(self.target_op_list)

    def callback(self, xk):
        print('Current iteration:', len(self.hist['x']), 'loss:', self.hist['loss'][-1])
        self.hist['x'].append(xk.tolist())
        self.hist['loss'].append(self.cost_func(xk))

    def run(self, target_op_list):
        self.target_op_list = target_op_list
        circuit = self.construct_circuit(target_op=target_op_list[0])

        initial_point = np.random.random(len(circuit.params_name))

        self.ham()

        self.hist = {'x': [initial_point.tolist()], 'loss': [self.cost_func(initial_point)]}
        start_time = time.time()
        res = minimize(self.cost_func, initial_point, method='L-BFGS-B', callback=self.callback, options={'maxiter':20})
        end_time = time.time()
        exec_time = end_time - start_time

        print(res)
        self.hist['execute time'] = exec_time
        return self.hist


if __name__ == "__main__":
    print("begin test:")
    num_latent, num_trash = 2, 2
    num_qubits = num_latent + num_trash
    qcae = QCAE(num_latent=num_latent, num_trash=num_trash)

    target_op_list = []
    for i in range(2):
        target_op = random_circuit(num_qubits, 10)

        target_op_list.append(target_op)
    qcae.run(target_op_list=target_op_list)
