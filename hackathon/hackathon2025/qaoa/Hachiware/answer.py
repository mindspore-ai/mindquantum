import numpy as np
import networkx as nx
from mindquantum import Circuit, X, Y, I, H, Z, S, CNOT, Hamiltonian, QubitOperator, Simulator
from itertools import combinations
import copy
from mindquantum.simulator import get_stabilizer_string, get_tableau_string
from collections import deque
from mindquantum.core.circuit import Circuit, UN
import random
from mindquantum.core.gates import Measure
from mindquantum.io import OpenQASM

def add_H_layer(nqubit):
    circ = Circuit()
    for i in range(nqubit):
        circ += H.on(i)
    return circ


def build_hamiltonian(nqubit, h, J):
    qubit_op = QubitOperator()
    for i in range(nqubit):
        if abs(h[i]) > 1e-10:
            qubit_op += QubitOperator(f'Z{i}', h[i])
    qubit_pairs = list(combinations(range(nqubit), 2))
    for i, j in qubit_pairs:
        if i >= j:
            continue
        if abs(J[i][j]) > 1e-10:
            qubit_op += QubitOperator(f'Z{i} Z{j}', J[i][j])
    return Hamiltonian(qubit_op)


def add_YZ_gate(q1, q2):
    c = Circuit()
    c += Z.on(q1)
    c += S.on(q1)

    c += H.on(q2)
    c += CNOT.on(q2, q1)
    c += Z.on(q1)

    c += H.on(q1)
    c += S.on(q1)
    c += H.on(q1)
    c += S.on(q1)
    c += S.on(q1)

    c += CNOT.on(q2, q1)
    c += S.on(q1)
    c += H.on(q2)

    return c


def gradient(inaqubit, W, aqubits_k, aqubits_j):
    lindex_k = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_k)
    lindex_j = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_j)
    sum_weights_k = np.sum(W[ll, inaqubit] for ll in lindex_k)
    sum_weights_j = np.sum(W[ll, inaqubit] for ll in lindex_j)
    return -sum_weights_k + sum_weights_j


def pos_max_grad(inaqubits, W, aqubits_k, aqubits_j):
    all_grads_k = [gradient(inaqubit, W, aqubits_k, aqubits_j) for inaqubit in inaqubits]
    all_grads_j = -1.0 * np.array(all_grads_k)

    pos_max_k = np.argmax(all_grads_k)
    pos_max_j = np.argmax(all_grads_j)

    if all_grads_k[pos_max_k] > all_grads_j[pos_max_j]:
        return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
    elif all_grads_k[pos_max_k] < all_grads_j[pos_max_j]:
        return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]
    else:
        return ("k", inaqubits[pos_max_k], all_grads_k[pos_max_k]) if np.random.choice([0, 1]) else \
            ("j", inaqubits[pos_max_j], all_grads_j[pos_max_j])



def x_gates_circuit(circuit, select_num, nqubit=None):
    bits = np.zeros(nqubit)

    for i in range(nqubit):
        if i in select_num:
            circuit += X.on(i)
            bits[i] = -1
        else:
            bits[i] = 1

    return circuit, bits


def greedy(circ0, ham, nqubit, Q_triu, select_num, repeat):
    J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
    h = np.diag(Q_triu)

    best_circ = copy.deepcopy(circ0)
    best_energy = 100
    best_depth = None
    sim = Simulator('stabilizer', nqubit)
    for ii in range(repeat):
        circ = Circuit(UN(I, nqubit))
        if select_num is None:
            break
        else:
            circ, bits = x_gates_circuit(circ, select_num, nqubit)

            while True:
                delta_E = (2 * J @ bits + h) * -2 * bits
                index = np.argmin(delta_E)
                bits[index] = -bits[index]
                if delta_E[index] < 0:
                    circ += X.on(int(index))
                else:
                    break

        sim.reset()
        exp = sim.get_expectation(ham, circ)
        energy = exp.real

        if energy < best_energy:
            best_energy = energy
            best_circ = copy.deepcopy(circ)
    return best_circ


def post_selection(circ, ham, nqubit, Q_triu):
    circ_b = copy.deepcopy(circ)
    expectation_b = Simulator('stabilizer', circ_b.n_qubits).get_expectation(ham, circ_b).real
    print(expectation_b)

    # 这个我们用 measure 进行处里
    measure_circuit = Circuit()
    for index in range(0, circ.n_qubits):
        measure_circuit += Measure().on(index)

    sim = Simulator('stabilizer', circ.n_qubits)
    sim.apply_circuit(circ)
    result = sim.sampling(measure_circuit, shots=200)
    output_dic = result.data
    select_num_best = []
    for key in output_dic:
        select_num = []
        # print(key)
        circ0 = Circuit()
        circ0 += Circuit(UN(I, nqubit))
        x0 = np.ones(nqubit)
        for index in range(circ.n_qubits):
            if key[index] == '1':
                circ0 += X.on(circ.n_qubits - index - 1)
                x0[circ.n_qubits - index - 1] = -1
                select_num.append(circ.n_qubits - index - 1)
                # circ0 += X.on(index)
                # x0[index] = -1

        # sim0 = Simulator('stabilizer', circ0.n_qubits)
        # output0 = sim0.get_expectation(ham, circ0).real
        print("当前的电路", x0)
        output0 = 0
        nq = Q_triu.shape[0]
        for i in range(nq):
            output0 += Q_triu[i, i] * x0[i]

        for j in range(nq):
            for k in range(j + 1, nq):
                if Q_triu[j, k] != 0:
                    output0 += 2 * Q_triu[j, k] * x0[j] * x0[k]
                    # ham += QubitOperator(f'Z{j} Z{k}',2*Q_triu[j,k])
        if output0 <= expectation_b:
            circ_b = copy.deepcopy(circ0)
            expectation_b = output0
            select_num_best = copy.deepcopy(select_num)
    if len(select_num_best) == 0:
        select_num_best = copy.deepcopy(select_num)

    return circ_b, select_num_best


def solve(nqubit, Q_triu):
    ham = build_hamiltonian(nqubit, np.diag(Q_triu).flatten(), 2 * (Q_triu - np.diag(np.diag(Q_triu))))

    h_0 = np.diag(Q_triu).flatten()
    J_0 = 2 * (Q_triu - np.diag(np.diag(Q_triu)))

    Q_sym = Q_triu + Q_triu.T - np.diag(np.diag(Q_triu))
    Q = nx.from_numpy_array(Q_sym)
    W = nx.adjacency_matrix(Q).toarray()

    best_circ = add_H_layer(nqubit)
    fqubit = 1
    best_circ += Z.on(1)

    sim = Simulator('stabilizer', nqubit)
    sim.apply_circuit(best_circ)

    active_qubits_k = []
    active_qubits_j = []
    inactive_qubits = list(range(nqubit))
    quantum_state = sim.get_qs()

    gate_positions = []
    for nn in range(nqubit - 1):
        if nn == 0:
            nonzero_indices = np.nonzero(W[:, fqubit])[0].astype(int).tolist()
            qpair = int(np.random.choice(nonzero_indices))
            active_qubits_j.append(qpair)
            active_qubits_k.append(fqubit)
            inactive_qubits.remove(qpair)
            inactive_qubits.remove(fqubit)
            qubits = (fqubit, qpair)
        else:
            aset, qpair, gra = pos_max_grad(inactive_qubits, W, active_qubits_k, active_qubits_j)
            qpair = int(qpair)
            if aset == "k":
                qubits = (qpair, fqubit)
                active_qubits_k.append(qpair)
            elif aset == "j":
                qubits = (qpair, active_qubits_j[0])
                active_qubits_j.append(qpair)

            inactive_qubits.remove(qpair)
        best_circ += add_YZ_gate(int(qubits[0]), int(qubits[1]))
        gate_positions.append(qubits)
    sim.reset()
    best_circ2, select_num = post_selection(best_circ, ham, nqubit, Q_triu)
    return greedy(best_circ2, ham, nqubit, Q_triu, select_num=select_num, repeat=2)
