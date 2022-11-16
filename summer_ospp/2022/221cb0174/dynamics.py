from mindquantum import Circuit, RX, RZ, H, X, Y, Z, S
from mindquantum.core.gates import SWAP
from mindquantum.core.gates import Measure
from mindquantum.simulator import Simulator
import numpy as np
from scipy import linalg

def make_circ_m(thetas, theta_i, theta_j, qcidx):
    assert qcidx in [0, 1]
    circ = Circuit()
    circ += H.on(0)
    circ += H.on(1)
    circ += H.on(2)
    circ += H.on(5)
    circ += H.on(6)
    
    apply_dic = {0: [RX(thetas[0]).on(3, 1)],
                 1: [RX(thetas[1]).on(4, 2)],
                 2: [X.on(4, 3), RZ(thetas[2]).on(4), X.on(4, 3)],
                 3: [RX(thetas[3]).on(3)],
                 4: [RX(thetas[4]).on(4)],
                 5: [X.on(4, 3), RZ(thetas[5]).on(4), X.on(4, 3)]
                }

    
    for i in range(6):
        if i == theta_i:
            if qcidx == 1:
                circ += X.on(0)
            if i in [0, 1]:
                circ += X.on(i+3, (0, i+1))
            elif i in [2, 5]:
                circ += Z.on(3, 0)
                circ += Z.on(4, 0)
            else:
                circ += X.on(i, 0)
            if qcidx == 1:
                circ += X.on(0)
        
        for i_op in apply_dic[i]:
            circ += i_op
            
    apply_dic = {0: [RX(thetas[0]).on(7, 5)],
                 1: [RX(thetas[1]).on(8, 6)],
                 2: [X.on(8, 7), RZ(thetas[2]).on(8), X.on(8, 7)],
                 3: [RX(thetas[3]).on(7)],
                 4: [RX(thetas[4]).on(8)],
                 5: [X.on(8, 7), RZ(thetas[5]).on(8), X.on(8, 7)]
                }
    
    for i in range(6):
        if i == theta_j:
            circ += X.on(0)
            if i in [0, 1]:
                circ += X.on(i+7, (0, i+5))
            elif i in [2, 5]:
                circ += Z.on(7, 0)
                circ += Z.on(8, 0)
            else:
                circ += X.on(i+4, 0)
            circ += X.on(0)
        for i_op in apply_dic[i]:
            circ += i_op
    circ += SWAP.on((3, 7), 0)
    circ += SWAP.on((4, 8), 0)
    circ += H.on(0)
    return circ

def psi_partial_thetai(thetas, theta_i, part):
    circ = Circuit()
    circ += H.on(0)
    assert part in ["real", "imag"]
    if part == "imag":
        circ += Z.on(0)
        circ += S.on(0)
    circ += H.on(1)
    circ += H.on(2)
    circ += H.on(5)
    circ += H.on(6)
    apply_dic = {0: [RX(thetas[0]).on(3, 1)],
                 1: [RX(thetas[1]).on(4, 2)],
                 2: [X.on(4, 3), RZ(thetas[2]).on(4), X.on(4, 3)],
                 3: [RX(thetas[3]).on(3)],
                 4: [RX(thetas[4]).on(4)],
                 5: [X.on(4, 3), RZ(thetas[5]).on(4), X.on(4, 3)]
                }

    for i in range(6):
        if i == theta_i:
            circ += X.on(0)
            if i in [0, 1]:
                circ += X.on(i+3, (0, i+1))
            elif i in [2, 5]:
                circ += Z.on(3, 0)
                circ += Z.on(4, 0)
            else:
                circ += X.on(i, 0)
            circ += X.on(0)
        
        for i_op in apply_dic[i]:
            circ += i_op
            
    apply_dic = {0: [RX(thetas[0]).on(7, 5)],
                 1: [RX(thetas[1]).on(8, 6)],
                 2: [X.on(8, 7), RZ(thetas[2]).on(8), X.on(8, 7)],
                 3: [RX(thetas[3]).on(7)],
                 4: [RX(thetas[4]).on(8)],
                 5: [X.on(8, 7), RZ(thetas[5]).on(8), X.on(8, 7)]
                }
    for i in range(6):
        for i_op in apply_dic[i]:
            circ += i_op

    return circ


def make_circ_xxzz(thetas, theta_i, oper_idx, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_diff_dic = {0: [X.on(7, 0)], 1: [X.on(8, 0)],
                      2: [Z.on(7, 0), Z.on(8, 0)]}
    if qcidx == 0:
        circ += X.on(0)
        
    for i_op in apply_diff_dic[oper_idx]:
        circ += i_op
    if qcidx == 0:
        circ += X.on(0)
    circ += SWAP.on([3, 7], 0)
    circ += SWAP.on([4, 8], 0)
    circ += H.on(0)
    return circ


def make_circ_xy1(thetas, theta_i, oper_idx, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_qubit = {0: 7, 1: 8}
    if qcidx == 0:
        circ += X.on(apply_qubit[oper_idx], 0)
        circ += Y.on(apply_qubit[oper_idx], 0)
    elif qcidx == 1:
        circ += X.on(0)
        circ += Y.on(apply_qubit[oper_idx], 0)
        circ += X.on(apply_qubit[oper_idx], 0)
        circ += X.on(0)
    circ += SWAP.on([3, 7], 0)
    circ += SWAP.on([4, 8], 0)
    circ += H.on(0)
    return circ

def make_circ_xx(thetas, theta_i, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "imag")
    apply_gates = {0: X.on(7),
                   1: Y.on(7),
                   2: X.on(8),
                   3: Y.on(8)}
    circ += apply_gates[qcidx]
    circ += SWAP.on([3, 7], 0)
    circ += SWAP.on([4, 8], 0)
    circ += H.on(0)
    return circ

def make_circ_xy(thetas, theta_i, qcidx):
    circ = psi_partial_thetai(thetas, theta_i, "real")
    apply_gates = {0: [X.on(7, 0), Y.on(7, 0)],
                   1: [Y.on(7, 0), X.on(7, 0)],
                   2: [X.on(8, 0), Y.on(8, 0)],
                   3: [Y.on(8, 0), X.on(8, 0)]}
    circ += X.on(0)
    circ += apply_gates[qcidx][0]
    circ += X.on(0)
    circ += apply_gates[qcidx][1]
    circ += SWAP.on([3, 7], 0)
    circ += SWAP.on([4, 8], 0)
    circ += H.on(0)
    return circ

def make_circ_rho(thetas, theta_i):
    circ = psi_partial_thetai(thetas, theta_i, "imag")
    circ += SWAP.on([3, 7], 0)
    circ += SWAP.on([4, 8], 0)
    circ += H.on(0)
    return circ

def measure_qc_statevec(qc):
    vec = qc.get_qs()
    vec = vec.reshape(-1, 2)
    res = np.einsum("ab, ab -> b", np.conj(vec), vec).real
    return res[0] - res[1]
    # qc += Measure('q0').on(0) 
    # sim = Simulator('projectq', 9)
    # sim.reset()
    # result = sim.sampling(qc, shots=100000) 
    # res = result.data
    # if len(res) == 2:
    #     return (res['0'] - res['1']) / (res['0'] + res['1'])
    # elif '0' in res.keys():
    #     return 1
    # elif '1' in res.keys():
    #     return -1

def measure_z0(thetas):
    circ = Circuit()
    circ += H.on(0)
    circ += H.on(1)
    apply_dic = {0: [RX(thetas[0]).on(2, 0)],
                 1: [RX(thetas[1]).on(3, 1)],
                 2: [X.on(3, 2), RZ(thetas[2]).on(3), X.on(3, 2)],
                 3: [RX(thetas[3]).on(2)],
                 4: [RX(thetas[4]).on(3)],
                 5: [X.on(3, 2), RZ(thetas[5]).on(3), X.on(3, 2)]
                }
    for i in range(6):
        for i_op in apply_dic[i]:
            circ += i_op
            
    vec = circ.get_qs()
    vec = vec.reshape(2, 2, 2, 2)
    res = np.einsum("abcd, abcd -> b", np.conj(vec), vec).real
    return res[0] - res[1]
    # circ += Measure('q2').on(2) 
    # sim = Simulator('projectq', 4)
    # sim.reset()
    # result = sim.sampling(circ, shots=100000) 
    # res = result.data
    # if len(res) == 2:
    #     return (res['0'] - res['1']) / (res['0'] + res['1'])
    # elif '0' in res.keys():
    #     return 1
    # elif '1' in res.keys():
    #     return -1


def get_grad(thetas):
    M = np.zeros((6, 6))
    for i in range(6):
        for j in range(i, 6):
            res = []
            qc = make_circ_m(thetas, i, j, 0)
            res.append(measure_qc_statevec(qc))
            qc = make_circ_m(thetas, i, j, 1)
            res.append(measure_qc_statevec(qc))
            M[i, j] = 2 * res[0] - 2 * res[1]
            M[j, i] = 2 * res[0] - 2 * res[1]
    V = []
    for i in range(6):
        res = 0
        for j in range(3):
            coeff = 0.25 if j == 2 else 1
            qc = make_circ_xxzz(thetas, i, j, 0)
            res = res - 2 * coeff * measure_qc_statevec(qc)
            qc = make_circ_xxzz(thetas, i, j, 1)
            res = res + 2 * coeff * measure_qc_statevec(qc)
        for j in range(2):
            qc = make_circ_xy1(thetas, i, j, 0)
            res = res - 2 * 0.25 * measure_qc_statevec(qc)
            qc = make_circ_xy1(thetas, i, j, 1)
            res = res - 2 * 0.25 * measure_qc_statevec(qc)
        for j in range(4):
            qc = make_circ_xx(thetas, i, j)
            res = res - 0.25 * 2 * measure_qc_statevec(qc)
        for j in range(4):
            coeff = (-1) ** j
            qc = make_circ_xy(thetas, i, j)
            res = res - 0.25 * 2 * coeff * measure_qc_statevec(qc)
        qc = make_circ_rho(thetas, i)
        res = res + 2 * measure_qc_statevec(qc)
        V.append(res)
    M = M + np.eye(6) * 1.e-8
    grad_vec = linalg.solve(M, V)
    return grad_vec


