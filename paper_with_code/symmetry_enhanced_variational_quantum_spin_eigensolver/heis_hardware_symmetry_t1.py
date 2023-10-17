import time
import numpy as np
import scipy as sp
import scipy.optimize as sopt
from mindquantum.simulator import Simulator
from mindquantum.core import CNOT, H, X, Z, RY, RZ, Circuit, QubitOperator, PhaseShift, Hamiltonian
import quimb as qu
import joblib


def revert_matrix(N):
    row = np.arange(2**N)
    col = np.zeros(2**N)
    data = np.ones(2**N)
    for i in range(2**N):
        bin_str = f'{{:0{N}b}}'.format(i)
        bin_str = bin_str[::-1]
        col[i] = int(bin_str, 2)
    mat = sp.sparse.csc_matrix((data, (row, col)), shape=(2**N, 2**N))
    return mat


def Heis_Ham(N, J):
    ham = QubitOperator()
    for i in range(N - 1):
        strength = J
        ham += QubitOperator(((i, 'X'), (i + 1, 'X')), strength)
        ham += QubitOperator(((i, 'Y'), (i + 1, 'Y')), strength)
        ham += QubitOperator(((i, 'Z'), (i + 1, 'Z')), strength)
    return Hamiltonian(ham)


def basis_creation(N):
    basis = []
    check_half = lambda s: sum([int(c) for c in s])
    for index in range(2**N):
        bin_str = f'{{:0{N}b}}'.format(index)
        tmp = check_half(bin_str)
        if tmp == N // 2:
            basis.append(qu.computational_state(bin_str, sparse=True))
    print(np.shape(basis))
    return sp.sparse.hstack(basis)


def basis_transform(obj, basis):
    if qu.isvec(obj):
        return basis @ obj
    else:
        return qu.dag(basis) @ obj @ basis


def get_eigstates(N):
    basis = basis_creation(N)
    ham = qu.ham_heis(N, j=1, cyclic=False)
    ham_reduce = basis_transform(ham, basis)
    eigstate_reduce = qu.eigvecsh(ham_reduce)
    # print(4 * qu.eigvalsh(ham_reduce))
    # print(4 * qu.eigvalsh(ham))
    eigstates = []
    for index in range(np.shape(eigstate_reduce)[1]):
        obj = qu.qu(eigstate_reduce[:, index], qtype='ket')
        obj = basis_transform(obj, basis)
        eigstates.append(obj)
    return eigstates

def N_block(params_str, wires, coeff=1):
    circuit = Circuit()
    circuit += RZ(-np.pi / 2).on(wires[1])

    circuit += CNOT.on(wires[0], wires[1])

    circuit += RZ({params_str: -2 * coeff}).on(wires[0])
    circuit += RZ(np.pi / 2).on(wires[0])

    circuit += RY({params_str: 2 * coeff}).on(wires[1])
    circuit += RY(-np.pi / 2).on(wires[1])

    circuit += CNOT.on(wires[1], wires[0])

    circuit += RY({params_str: -2 * coeff}).on(wires[1])
    circuit += RY(np.pi / 2).on(wires[1])

    circuit += CNOT.on(wires[0], wires[1])

    circuit += RZ(np.pi / 2).on(wires[0])
    return circuit


def ansatz(prefix, N, num_layer, sz):
    ansatz = Circuit()
    params_index = 0

    for i in range(N):
        if i != N // 2:
            ansatz += X.on(i)
    for i in range(0, N, 2):
        ansatz += H.on(i)
        ansatz += CNOT.on(i + 1, i)

    for layer_index in range(num_layer):
        for i in range(0, N, 2):
            ansatz += N_block(prefix + str(params_index), [i, i + 1])
            params_index += 1

        for i in range(1, N - 1, 2):
            ansatz += N_block(prefix + str(params_index), [i, i + 1])
            params_index += 1

    return ansatz


def train_func(params, grad_op, gs=None, sim=None, circ=None, rev_mat=None):

    global loss_hist
    global param_hist
    global fids_hist
    global iteration

    f, g = grad_op(params)
    g1 = np.array(g[0, 0, :].real)
    f1 = f[0, 0].real

    if gs is not None:
        sim.apply_circuit(circ, params)
        state = rev_mat @ qu.qu(sim.get_qs(), qtype='ket')

        fid = qu.fidelity(state, gs, squared=True)
        fids_hist.append(fid)
        sim.reset()

    if iteration % 20 == 0:
        print('iteration: {}, expt: {:.3f}, fid: {:.3f}'.format(
            iteration, f1, fids_hist[-1]))

    iteration += 1
    loss_hist.append(f1)
    param_hist.append(params)

    return f1, g1


loss_hist = []
param_hist = []
fids_hist = []
iteration = 0

if __name__ == '__main__':
    prefix = 'p_'
    N = 4
    num_layer = 2
    J = 1
    sz = 0

    rev_mat = revert_matrix(N)

    circ = ansatz(prefix, N, num_layer, sz).as_ansatz()
    ham = Heis_Ham(N, J)

    ham_matrix = qu.ham_heis(N, J, cyclic=False)
    print(qu.eigvalsh(ham_matrix) * 4)
    eigenstates = get_eigstates(N)
    target_state = eigenstates[1]
    print(target_state)

    np.random.seed()
    init_params = (np.random.rand(len(circ.params_name)) - .5) * np.pi

    sim = Simulator('mqvector', N)
    grad_op = sim.get_expectation_with_grad(
        ham, circ)

    res = sopt.minimize(train_func,
                        init_params,
                        args=(
                            grad_op,
                            target_state,
                            sim,
                            circ,
                            rev_mat,
                        ),
                        jac=True,
                        method='Nelder-Mead',
                        tol=1e-10,
                        options={'disp': False})

    print('N={}, layer={}, iteration: {}, expt: {:.3f}, fid: {:.3f}'.format(
        N, num_layer, iteration, loss_hist[-1], fids_hist[-1]))
    print(param_hist[-1])
    # joblib.dump(loss_hist,
    #             'Heis_expt_N={}_layer={}_J={}'.format(N, num_layer, J))
    # joblib.dump(param_hist,
    #             'Heis_params_N={}_layer={}_J={}'.format(N, num_layer, J))
