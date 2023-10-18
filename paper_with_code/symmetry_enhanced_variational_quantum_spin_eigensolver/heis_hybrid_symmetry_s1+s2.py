import time
import numpy as np
import scipy as sp
import scipy.optimize as sopt
from mindquantum.simulator import Simulator
from mindquantum.core import CNOT, H, X, Z, RY, RZ, Circuit, QubitOperator, PhaseShift, Hamiltonian
import quimb as qu
import joblib
import multiprocessing as mp


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


def s_alpha_mq(N, alpha: str = 'X'):
    alpha = alpha.upper()
    out = QubitOperator()
    for i in range(N):
        out += QubitOperator(alpha + f'{i}')
    return out * 0.5


def s_tot_op(N):
    s_x_op = s_alpha_mq(N, 'X')**2
    s_y_op = s_alpha_mq(N, 'Y')**2
    s_z_op = s_alpha_mq(N, 'Z')**2
    op = s_x_op + s_y_op + s_z_op
    op.compress()
    return op


def Heis_Ham(N, J):
    ham = QubitOperator()
    for i in range(N - 1):
        strength = J
        ham += QubitOperator(((i, 'X'), (i + 1, 'X')), strength)
        ham += QubitOperator(((i, 'Y'), (i + 1, 'Y')), strength)
        ham += QubitOperator(((i, 'Z'), (i + 1, 'Z')), strength)
    return Hamiltonian(ham)


def basis_creation(N, sz=0):
    basis = []
    check_half = lambda s: sum([int(c) for c in s])
    for index in range(2**N):
        bin_str = f'{{:0{N}b}}'.format(index)
        tmp = check_half(bin_str)
        if N // 2 - tmp == sz:
            basis.append(qu.computational_state(bin_str, sparse=True))
    # print(np.shape(basis))
    return sp.sparse.hstack(basis)


def basis_transform(obj, basis):
    if qu.isvec(obj):
        return basis @ obj
    else:
        return qu.dag(basis) @ obj @ basis


def get_eigstates(N, sz=0):
    basis = basis_creation(N, sz)
    ham = qu.ham_heis(N, j=1, cyclic=False)
    ham_reduce = basis_transform(ham, basis)
    eigstate_reduce = qu.eigvecsh(ham_reduce)
    print(4 * qu.eigvalsh(ham_reduce)[:8])
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


def ansatz(prefix, N, num_layer, init_str):
    ansatz = Circuit()
    params_index = 0

    for index, c in enumerate(init_str):
        if c == '1':
            ansatz += X.on(index)

    for layer_index in range(num_layer):
        for i in range(0, N, 2):
            ansatz += N_block(prefix + str(params_index), [i, i + 1])
            params_index += 1

        for i in range(1, N - 1, 2):
            ansatz += N_block(prefix + str(params_index), [i, i + 1])
            params_index += 1

        for i in range(N):
            ansatz += PhaseShift(params_index).on(i)
            params_index += 1

    return ansatz


def train_func(N,
               num_layer,
               run,
               ops,
               init_strs,
               beta,
               weights=[2, 1],
               gs=None,
               rev_mat=None,
               J=1,
               prefix='p_'):

    loss_hist = []
    param_hist = []
    fids_hist = []
    iteration = 0

    def value_and_grad(params, sim, grad_ops, circs, beta, weights, gs, rev_mat):
        nonlocal loss_hist
        nonlocal param_hist
        nonlocal fids_hist
        nonlocal iteration
        fids = []
        cost, cost_gn = [], []
        for init_index in range(len(grad_ops)):
            f, g = grad_ops[init_index](params)
            g1, g2= np.array(g[0, 0, :].real), np.array(g[0, 1, :].real)
            f1, f2 = f[0, 0].real, f[0, 1].real
            cost.append([f1, beta * (f2 ** 2)])
            cost_gn.append([g1, beta * 2 * g2])

            if gs is not None:
                sim.apply_circuit(circs[init_index], params)
                state = rev_mat @ qu.qu(sim.get_qs(), qtype='ket')
                fid = qu.fidelity(state, gs[init_index], squared=True)
                fids.append(fid)
                sim.reset()
        fids_hist.append(fids)

        if iteration % 20 == 0:
            print('iteration: {}, expt: {}, fid: {}'.format(
                iteration, [tmp[0] for tmp in cost], fids_hist[-1]))

        iteration += 1
        loss_hist.append(cost)
        param_hist.append(params)

        final_cost = sum([w * sum(c) for w, c in zip(weights, cost)])
        final_cost_gn = sum([w * sum(c_gn) for w, c_gn in zip(weights, cost_gn)])

        return final_cost, final_cost_gn

    np.random.seed()
    sim = Simulator('mqvector', N)

    for layer in range(1, num_layer + 1):
        loss_hist = []
        param_hist = []
        fids_hist = []
        iteration = 0

        circs = [
            ansatz(prefix, N, layer, init_str=init_str).as_ansatz()
            for init_str in init_strs
        ]

        if layer == 1:
            init_params = np.random.rand(len(circs[0].params_name)) - .5
        else:
            init_params = joblib.load(
                'data/hardware_sym_s1+s2_Heis_params_run={}_N={}_layer={}_J={}'
                .format(run, N, layer - 1, J))[-1]
            init_params = np.concatenate([
                init_params, init_params[-len(circs[0].params_name) // layer:]
            ])

        grad_ops = [sim.get_expectation_with_grad(ops, circ) for circ in circs]

        res = sopt.minimize(value_and_grad,
                            init_params,
                            args=(
                                sim,
                                grad_ops,
                                circs,
                                beta,
                                weights,
                                target_state,
                                rev_mat,
                            ),
                            jac=True,
                            method='l-bfgs-b',
                            tol=1e-10,
                            options={'disp': False})

        print(
            'run = {} N={}, layer={}, iteration: {}, expt: {}, fid: {}'.format(
                run, N, layer, iteration, [tmp[0] for tmp in loss_hist[-1]], fids_hist[-1]))
        joblib.dump(
            loss_hist,
            'data/hardware_sym_s1+s2_Heis_expt_run={}_N={}_layer={}_J={}'.
            format(run, N, layer, J))
        joblib.dump(
            param_hist,
            'data/hardware_sym_s1+s2_Heis_params_run={}_N={}_layer={}_J={}'.
            format(run, N, layer, J))
        joblib.dump(
            fids_hist,
            'data/hardware_sym_s1+s2_Heis_fids_run={}_N={}_layer={}_J={}'.
            format(run, N, layer, J))


if __name__ == '__main__':
    prefix = 'p_'
    N = 8
    num_layer = 22
    J = 1
    sz = 0
    runs = 1
    targ_val = 0
    beta = 10

    rev_mat = revert_matrix(N)

    init_strs = ['01' * (N // 2), '10' * (N // 2)]

    ham = Heis_Ham(N, J)

    stot = s_tot_op(N)
    pen_1 = stot
    pen_1.compress()

    pen_1 = Hamiltonian(pen_1).sparse(N)
    pen_1.get_cpp_obj()

    ops = [ham, pen_1]

    eigenstates = get_eigstates(N, sz)
    target_state = [eigenstates[0], eigenstates[3]]

    tasks = []
    for run in range(1, runs + 1):
        proc = mp.Process(target=train_func,
                          args=(N, num_layer, run, ops, init_strs, beta, [2, 1],
                                target_state, rev_mat))
        tasks.append(proc)
    [t.start() for t in tasks]
    [t.join() for t in tasks]
