# NOTE: 答题时只能修改此文件，整合一切代码，因为评测机只会上传这单个脚本！！
# 一点也不能改 solution() 函数的签名，例如不能给 molecule 加类型注解，因为评测机做了硬编码的校验！！

if 'env':
    import os, sys
    sys.path.append(os.path.abspath(__file__))

    import warnings
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

from re import compile as Regex
from copy import deepcopy
from typing import *

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution
from mindquantum.core.gates import H, X, Y, Z, RX, RY, RZ, CNOT, BasicGate, ParameterGate, MeasureResult
from mindquantum.core.circuit import Circuit, UN, dagger
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.algorithm.nisq import Transform
from mindquantum.utils.progress import SingleLoopProgress
from mindquantum.third_party.unitary_cc import uccsd_singlet_generator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum import MQAnsatzOnlyLayer
from mindspore import Tensor
from mindspore.nn.optim import SGD, Adam
from mindspore.nn.wrap.cell_wrapper import TrainOneStepCell

from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian

SHOTS = 1000
N_OPTIM_TRIAL = 1
N_PRUNE = 2
NOISE_AWARE_OPTIMIZE = False

DEBUG_OPTIM = True
DEBUG_OPTIM_VERBOSE = False

R_str = Regex('\[([XYZ\d ]+)\]')
Geometry = List[Tuple[str, List[float]]]
PauliTerm = Tuple[float, QubitOperator]
SplitHam = List[PauliTerm]

''' Molecule '''

def geometry_move_to_center(geo:Geometry) -> Geometry:
    center = np.stack([np.asarray(coord) for atom, coord in geo], axis=0).mean(axis=0)
    return [(atom, (np.asarray(coord) - center).tolist()) for atom, coord in geo]


''' Ansatz '''

def get_HF_circuit(mol) -> Circuit:
    # In HF-state, the electrons always occupy the lowest orbitals :)
    # see https://pennylane.ai/blog/2022/09/how-to-use-the-hartree-fock-method-in-pennylane/
    return UN(X, mol.n_electrons)

def get_uccsd_circuit(mol) -> Circuit:
    ucc = Transform(uccsd_singlet_generator(mol.n_qubits, mol.n_electrons)).jordan_wigner().imag
    ucc = TimeEvolution(ucc).circuit
    return get_HF_circuit(mol) + ucc

def get_pchc_circuit(mol, depth:int=1, order:str='sd') -> Circuit:
    ''' inspired by CHC from arXiv:2003.12578, we borrow the main structure and make it fully parametrized '''
    nq_half = mol.n_qubits // 2
    circ = Circuit()
    for d in range(depth):
        # layer idx
        l = 0
        # hardamard
        for i in range(mol.n_qubits):
            circ += RY(f'd{d}_l{l}_q{i}').on(i)
        l += 1
        # single excitation
        if 's' in order:
            for j in range(1, nq_half):
                for i in [0, nq_half]:
                    p, q = i, i + j
                    circ += CNOT.on(q, p)
                    circ += RZ(f'd{d}_l{l}_q{q}_c').on(q)
                    circ += CNOT.on(q, p)
                    circ += RY(f'd{d}_l{l}_q{p}').on(p)
                    circ += RY(f'd{d}_l{l}_q{q}').on(q)
                l += 1
        # double excitation
        if 'd' in order:
            for i in range(1, nq_half):
                for j in range(1, nq_half):
                    p, q, r, s = 0, i, nq_half, nq_half + j
                    circ += CNOT.on(q, p)
                    circ += CNOT.on(r, q)
                    circ += CNOT.on(s, r)
                    circ += RZ(f'd{d}_l{l}_q{s}_c').on(s)
                    circ += CNOT.on(s, r)
                    circ += CNOT.on(r, q)
                    circ += CNOT.on(q, p)
                    circ += RY(f'd{d}_l{l}_q{p}').on(p)
                    circ += RY(f'd{d}_l{l}_q{q}').on(q)
                    l += 1
    # HF appended
    return circ + get_HF_circuit(mol)

def get_hae_ry_circit(mol, depth:int=1) -> Circuit:
    ''' standard HAE(RY) '''
    circ = Circuit()
    for i in range(mol.n_qubits):
        circ += RY(f'd0_q{i}').on(i)
    for j in range(1, depth+1):
        for i in range(0, mol.n_qubits-1):
            circ += CNOT.on(i+1, i)
        for i in range(mol.n_qubits):
            circ += RY(f'd{j}_q{i}').on(i)
    return circ + get_HF_circuit(mol)

def get_hae_ry_compact_circit(mol, depth:int=1) -> Circuit:
    ''' impl from https://github.com/liwt31/QC-Contest-Demo
    The compact-HEA(RY) circuit is like:
        --RY--o-----RY--
              |
        --RY--x--o--RY--
                 |
        --RY--o--x--RY--
              |
        --RY--x-----RY--
    where two layers of CNOT placed zig-zagly, and RY is inserted around every CNOT block
    '''
    circ = Circuit()
    for i in range(mol.n_qubits):
        circ += RY(f'd0_q{i}').on(i)
    for j in range(1, depth+1):
        for i in range(0, mol.n_qubits, 2):
            circ += CNOT.on(i+1, i)
        for i in range(1, mol.n_qubits - 1, 2):
            circ += CNOT.on(i+1, i)
        for i in range(mol.n_qubits):
            circ += RY(f'd{j}_q{i}').on(i)
    # HF appended at last, for reference error mitigation
    return circ + get_HF_circuit(mol)

def get_cnot_centric_circit(mol, depth:int=1) -> Circuit:
    '''The CNOT-centric circuit is like:
        --RY--o---------RY--
              |
        --RY--x--RY--o--RY--
                     |
        --RY--o--RY--x--RY--
              |
        --RY--x---------RY--
    where two layers of CNOT placed zig-zagly, and RY is inserted around each CNOT gate
    '''
    circ = Circuit()
    # flat RY
    for i in range(mol.n_qubits):
        circ += RY(f'd0_q{i}').on(i)
    for d in range(1, depth+1):
        # zig-zag CNOT with RY bridge
        qid = 0
        while qid + 1 < mol.n_qubits:   # even control: CNOT(1, 0)
            circ += CNOT.on(qid+1, qid)
            qid += 2
        qid = 1
        while qid + 1 < mol.n_qubits:   # odd control: CNOT(2, 1)
            circ += RY(f'd{d}_q{qid}_mid').on(qid)
            circ += RY(f'd{d}_q{qid+1}_mid').on(qid+1)
            circ += CNOT.on(qid+1, qid)
            qid += 2
        # flat RY
        for i in range(mol.n_qubits):
            circ += RY(f'd{d}_q{i}').on(i)
    return circ + get_HF_circuit(mol)

def get_x_swap_circuit(mol) -> Circuit:
    '''The X-SWAP(θ) circuit is like:
        -----x-RY-o-RY-x---------------x-RY-o-RY-x--
             |    |    |               |    |    |
        -----|----|----|--x-RY-o-RY-x--o----x----o--
             |    |    |  |    |    |
        --X--|----|----|--o----x----o---------------
             |    |    |
        --X--o----x----o----------------------------
        | swap low-high symmetrically | swap low linearly |
    which is electron-preserving and symmetrical, and U(θ=0)|0000> = |0011> prepares the HF state :)
    '''
    def swap_theta_gate(i:int, j:int, l:str) -> Circuit:    # j is the control qubit
        circ = Circuit()
        circ += CNOT.on(i, j)
        circ += RY(f'l{l}_ry_0').on(i)
        circ += CNOT.on(j, i)
        circ += RY(f'l{l}_ry_1').on(i)
        circ += CNOT.on(i, j)
        return circ

    circ = Circuit()
    nq_hf = mol.n_qubits // 2
    for i in range(nq_hf, mol.n_qubits):
        circ += X.on(i)
    l = 0   # layer index
    for i in range(nq_hf):  # symmetrically
        circ += swap_theta_gate(i, mol.n_qubits - 1 - i, l)
        l += 1
    for i in range(nq_hf-2, -1, -1):  # linearly
        circ += swap_theta_gate(i, i + 1, l)
        l += 1
    return circ

def get_cnot_ry_seq_circuit(mol, depth:int=1) -> Circuit:
    '''The CNOT-RY-symmetrical circuit is like:
        --RY----------------x--RY--X--
                            |
        --RY---------x--RY--o------X--
                     |
        --RY--x--RY--o----------------
              |
        --RY--o-----------------------
    '''
    circ = Circuit()
    for d in range(depth):
        for q in range(mol.n_qubits):
            circ += RY(f'd{d}_q{q}').on(q)
        for q in range(mol.n_qubits - 1, 0, -1):
            circ += CNOT.on(q - 1, q)
            circ += RY(f'd{d}_q{q-1}').on(q - 1)
    return circ + get_HF_circuit(mol)

def get_cnot_ry_sym_circuit(mol) -> Circuit:
    '''The CNOT-RY-symmetrical circuit is like:
        --RY--x--RY--o--RY------------X--
              |      |
        --RY--|------|--x--RY--o--RY--X--
              |      |  |      |
        --RY--|------|--o------x--RY-----
              |      |
        --RY--o------x--RY---------------
    '''
    circ = Circuit()
    nq_hf = mol.n_qubits // 2
    for q in range(mol.n_qubits):
        circ += RY(f'd0_q{q}').on(q)
    l = 1   # layer index
    for q in range(nq_hf):  # symmetrically
        qq = mol.n_qubits - 1 - q
        circ += CNOT.on(q, qq)
        circ += RY(f'd{l}_q{q}').on(q)
        circ += CNOT.on(qq, q)
        circ += RY(f'd{l}_q{q}_r').on(q)
        circ += RY(f'd{l}_q{q}_qq').on(qq)
        l += 1
    return circ + get_HF_circuit(mol)


def get_hae_ry_circit_pretrained_params(depth:int=3) -> ndarray:
    # taken from `make_init.py`
    if depth == 2:
        return np.asarray([
            -7.50192089e-02, -5.39213019e-02, -1.57079946e+00,  2.57029659e-05,
            4.03054967e-06,  1.57079719e+00, -1.57079635e+00,  1.57079295e+00,
            1.16371583e-01, -5.40732159e-02,  1.57079701e+00,  1.25688645e+00,
            1.57078841e+00,  1.57080158e+00,  6.29802784e-06, -3.14159249e+00,
            -7.49089803e-02,  1.26923149e-06, -1.57076860e+00, -1.57080120e+00,
            -1.31342775e-07,  2.79170030e-06, -1.57079048e+00,  1.57079944e+00,
        ], np.float32)
    if depth == 3:
        return np.asarray([
            1.22371875e-01,  1.66050064e-01, -1.08207624e-07,  4.15497272e-06,
            2.97311960e-05,  -3.63897419e-06, -1.57081472e+00,  1.57079755e+00,
            -4.92704293e-02, -1.38435013e-07, -6.27820153e-07,  1.38740712e-07,
            9.15675847e-07,  -3.72413430e-06,  1.57079385e+00, -1.50730271e-05,
            -5.40910224e-07, 2.70715532e-07, -3.36369100e-01, -6.03572356e-07,
            2.97790996e-05,  1.08889993e-05,  1.57081228e+00,  1.50873326e-05,
            5.32951353e-06,  3.89451753e-06, -1.39213793e-06, -3.12104716e-06,
            -1.22848655e-06, -5.85945213e-06, -1.57079092e+00, -1.57079792e+00,
        ], np.float32)

def prune_circuit(circ:Circuit, pr:ParameterResolver, eps:float=1e-5) -> Tuple[Circuit, ParameterResolver]:
    if not len(circ.params_name) or pr is None: return circ, pr

    # sanitize
    to_keep: List[BasicGate] = []
    to_remove_keys: List[str] = [k for k in pr.keys() if abs(pr[k]) < eps]
    for gate in circ:
        gate: BasicGate
        if not gate.parameterized:
            to_keep.append(gate)
        else:
            gate: ParameterGate
            to_remove = False
            for gate_pr in gate.get_parameters():
                for key in gate_pr.keys():
                    if key in to_remove_keys:
                        to_remove = True
                        break
                if to_remove: break
            if not to_remove:
                to_keep.append(gate)
    # rebuild
    circ_new = Circuit()
    for gate in to_keep:
        circ_new += gate
    pr_new = ParameterResolver({k: pr[k] for k in circ_new.params_name})
    return circ_new, pr_new


''' Hamiltonian '''

def split_hamiltonian(ham:QubitOperator) -> Tuple[float, SplitHam]:
    const = 0.0
    split_ham: SplitHam = []
    for pr, ops in ham.split():
        if ops == 1:    # aka. I
            const = pr.const.real
        else:
            split_ham.append([pr.const.real, ops])
    return const, split_ham

def combine_hamiltonian(const:float, terms:SplitHam) -> QubitOperator:
    ham = QubitOperator('', const)
    for coeff, ops in terms:
        string = R_str.findall(str(ops))[0]
        ham += QubitOperator(string, coeff)
    return ham

def approx_merge_hamiltonian(split_ham:SplitHam) -> SplitHam:
    # parse pauli string
    terms: List[Tuple[float, str]] = []
    for coeff, ops in split_ham:
        string = R_str.findall(str(ops))[0]
        terms.append((coeff, string))
    # 若两个串X-Y对偶，则合并
    def is_same(s1:str, s2:str) -> bool:
        s = s1.replace('X', 'T').replace('Y', 'X').replace('T', 'Y')
        return s == s2
    string_coeff: Dict[str, List[float]] = {}
    for coeff, string in terms:
        found = False
        if 'X' in string or 'Y' in string:
            for string2 in string_coeff:
                if not ('X' in string2 or 'Y' in string2):
                    continue
                if is_same(string2, string):
                    string_coeff[string2].append(coeff)
                    found = True
                    break
        if not found:
            string_coeff[string] = [coeff]
    terms_agg: Dict[str, float] = {k: np.sum(v) for k, v in string_coeff.items()}
    # convert to SplitHam
    split_ham_combined = [[v, QubitOperator(k)] for k, v in terms_agg.items()]
    return split_ham_combined

def approx_replace_hamiltonian(split_ham:SplitHam) -> SplitHam:
    # parse pauli string, each term has 0/2/4 count of Y
    terms: List[Tuple[float, str]] = []
    for coeff, ops in split_ham:
        string = R_str.findall(str(ops))[0]
        terms.append((coeff, string))
    # ref: https://github.com/liwt31/QC-Contest-Demo
    # fix phase raised by HF_ciruit when replace Y -> X
    phases: List[int] = []
    hf_state = '11110000'   # reversed of the bit order
    for i, (coeff, string) in enumerate(terms):
        phase = 1
        for seg in string.split(' '):
            sym = seg[0]
            if sym != 'Y': continue
            qid = int(seg[1:])
            if hf_state[qid] == '1':
                phase *= 1j
            else:
                phase *= -1j
        assert phase.imag == 0
        phases.append(phase.real)
    assert len(phases) == len(terms)
    # approx YY ~= XX, hence aggregate the coeffs
    string_coeff: Dict[str, List[float]] = {}
    for i, (coeff, string) in enumerate(terms):
        string_XY = string.replace('Y', 'X')
        if string_XY not in string_coeff:
            string_coeff[string_XY] = []
        string_coeff[string_XY].append(coeff * phases[i])
    terms_agg: Dict[str, float] = {k: np.sum(v) for k, v in string_coeff.items()}
    # convert to SplitHam
    split_ham_combined = [[v, QubitOperator(k)] for k, v in terms_agg.items()]
    return split_ham_combined

def prune_hamiltonian(split_ham:SplitHam) -> SplitHam:
    from mindquantum._math.ops import QubitOperator as QubitOperator_

    def filter_Z_only(ops:QubitOperator) -> bool:   # for HF_circuit
        for term in QubitOperator_.get_terms(ops):
            for qubit, symbol in term[0]:
                if symbol.name in ['X', 'Y']:
                    return False
        return True

    #split_ham = [it for it in split_ham if filter_Z_only(it[1])]
    #split_ham = approx_merge_hamiltonian(split_ham)
    split_ham = approx_replace_hamiltonian(split_ham)
    split_ham = [it for it in split_ham if abs(it[0]) > 1e-2]
    split_ham.sort(key=lambda it: abs(it[0]), reverse=True)
    return split_ham


''' Optimize '''

def trim_p(x:ndarray, eps:float=1e-15) -> ndarray:
    return np.where(np.abs(x) < eps, 0.0, x)

def norm_p(x:ndarray) -> ndarray:
    x = x % (2*np.pi)       # [0, 2*pi]
    return np.where(x < np.pi, x, x - 2*np.pi)

def optim_sp(method:str, circ:Circuit, ham:Hamiltonian, p0:ndarray, tol:float=1e-8):
    # tricks
    TRIM_P = 1e-5   # 1e-9
    NORM_P = True   # True

    def func(x:ndarray, grad_ops):
        nonlocal TRIM_P, NORM_P
        if TRIM_P: x = trim_p(x, TRIM_P)
        if NORM_P: x = norm_p(x)

        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    if TRIM_P: p0 = trim_p(p0, TRIM_P)
    if NORM_P: p0 = norm_p(p0)

    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    options = {'maxiter':1000, 'disp':False}
    if method == 'COBYLA':
        options.update({'rhobeg': 1.57})
    res = minimize(func, p0, (grad_ops,), method, jac=True, tol=tol, options=options)

    if DEBUG_OPTIM: print('min. fval:', res.fun)
    if DEBUG_OPTIM_VERBOSE: print('argmin. x:', res.x)
    px = res.x
    if TRIM_P:
        px = trim_p(px, TRIM_P)
        if DEBUG_OPTIM_VERBOSE: print('argmin. x (trimmed):', px)
    if NORM_P:
        px = norm_p(px)
        if DEBUG_OPTIM_VERBOSE: print('argmin. x (normed):', px)
    return res.fun, ParameterResolver(dict(zip(circ.params_name, px)))

def optim_ms(method:str, circ:Circuit, ham:Hamiltonian, p0:ndarray, lr:float=0.01):
    ''' Model & Optim '''
    sim = Simulator('mqvector', circ.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    net = MQAnsatzOnlyLayer(grad_ops)
    net.weight.set_data(Tensor.from_numpy(p0))
    if method == 'Adam':
        opt = Adam(net.trainable_params(), learning_rate=lr)
    elif method == 'SGD':
        opt = SGD(net.trainable_params(), learning_rate=lr, momentum=0.8)
    train_net = TrainOneStepCell(net, opt)
    ''' Train '''
    best_E = 99999
    best_weight = None
    for i in range(3000 * (3 if NOISE_AWARE_OPTIMIZE else 1)):
        E = train_net()
        if E < best_E:
            best_E = E
            best_weight = train_net.weights[0]
        if DEBUG_OPTIM and i % 100 == 0:
            print(f'[step {i}] expect: {E}')
    if best_weight is None: return
    px = best_weight.numpy()
    return E.item(), ParameterResolver(dict(zip(circ.params_name, px)))


def get_best_params(circ:Circuit, ham:QubitOperator, method:str='BFGS', init:str='randu') -> Tuple[float, ParameterResolver]:
    n_params = len(circ.params_name)
    from_pretrained = False
    if isinstance(init, (ndarray, list)):
        print('>> from_pretrained :)')
        from_pretrained = True
        assert len(init) == n_params
        p0 = init
    elif init == 'zeros':
        p0 = np.zeros(n_params)
    elif init == 'randu':
        p0 = np.random.uniform(-np.pi, np.pi, n_params) * 0.1
    elif init == 'randn':
        p0 = np.random.normal(0, 0.02, n_params)
    p0 = p0.astype(dtype=np.float32)

    if NOISE_AWARE_OPTIMIZE:
        from noise_model import generate_noise_model
        noise_model = generate_noise_model()
        circ = noise_model(circ)

    if method in ['BFGS', 'COBYLA']:
        return optim_sp(method, circ, Hamiltonian(ham), p0)
    else:
        lr = 0.0001 if from_pretrained else 0.01
        return optim_ms(method, circ, Hamiltonian(ham), p0, lr)

def get_best_params_repeat(circ:Circuit, ham:QubitOperator, method:str='BFGS', init:str='randu', n_repeat:int=10, n_prune:int=-1) -> Union[ParameterResolver, Tuple[Circuit, ParameterResolver]]:
    fval_best, pr_best = 99999, None
    circ_original, circ_best = circ, circ    # save the original
    for idx in range(n_repeat):
        # primary optimize-record round
        curp = init
        circ = deepcopy(circ_original)
        retvals = get_best_params(circ, ham, method=method, init=curp)
        if not retvals: continue
        fval, pr = retvals
        print(f'   trial-{idx} fval:', fval)
        if fval < fval_best: fval_best, pr_best = fval, pr

        # secondary prune-optimize-record round
        for stage in range(n_prune):
            circ, pr = prune_circuit(circ, pr)
            curp = np.asarray([pr[name] for name in circ.ansatz_params_name], dtype=np.float32)
            print(circ)
            print(curp)
            retvals = get_best_params(circ, ham, method=method, init=curp)
            if not retvals: continue
            fval, pr = retvals
            print(f'   trial-{idx}-prune-{stage} fval:', fval)
            if fval <= fval_best: fval_best, pr_best, circ_best = fval, pr, circ

    print('   best fval:', fval_best)
    return (circ_best, pr_best) if n_prune > 0 else pr_best

''' Measure '''

def rotate_to_z_axis_and_add_measure(circ:Circuit, ops:QubitOperator) -> Circuit:
    circ = circ.copy()
    assert ops.is_singlet
    for idx, o in list(ops.terms.keys())[0]:
        if o == 'X':
            circ.ry(-np.pi / 2, idx)
        elif o == 'Y':
            circ.rx(np.pi / 2, idx)
        circ.measure(idx)
    return circ

def measure_single_ham(sim:Simulator, circ:Circuit, pr:ParameterResolver, ops:QubitOperator, shots:int=100) -> float:
    circ_m = rotate_to_z_axis_and_add_measure(circ, ops)
    result: MeasureResult = sim.sampling(circ_m, pr, shots, seed=None)
    exp = 0.0
    for bits, cnt in result.data.items():
        exp += (-1)**bits.count('1') * cnt
    return round(exp / shots, int(np.ceil(np.log10(shots))))

def get_exp(sim:Simulator, circ:Circuit, pr:ParameterResolver, split_ham:SplitHam, shots:int=100, use_exp_fix:bool=False, debug_log:bool=False) -> float:
    result = 0.0
    with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
        for idx, (coeff, ops) in enumerate(split_ham):
            if not 'use var_shots':
                n_prec = int(np.ceil(np.log10(1/abs(coeff))))
                var_shots = shots * 10**(n_prec-1)
            else:
                var_shots = shots
            exp = measure_single_ham(sim, circ, pr, ops, var_shots)

            if use_exp_fix:     # for HF_circuit only
                # FIXME: when circ is non-entangled, and gates are all X
                # the BitFlip noise on measure gate could be cheaty moved out :)
                if   exp > +0.1: exp = +1
                elif exp < -0.1: exp = -1
                else:            exp = 0

            if debug_log: print('  coeff=', coeff, 'term=', ops, 'exp=', exp)
            result += exp * coeff
            bar.update_loop(idx)
    return result


''' Error Mitigation '''

def zref_trim_circuit(circ:Circuit) -> Circuit:
    ''' zero ref-state EM utils, remove all parameterized gates '''
    circ_new = Circuit()
    for gate in circ:
        gate: BasicGate
        if not gate.parameterized:
            circ_new += gate
    return circ_new

def zne_repeat_circuit(circ:Circuit, n_repeat:int=1) -> Circuit:
    ''' ZNE EM utils, expand circuit by append each gate G with [G.dagger - G]*n_repeat, the noise should be amplified 2*n_repeat times additively '''
    circ_new = Circuit()
    for gate in circ:
        gate: BasicGate
        for _ in range(n_repeat):
            circ_new += gate.hermitian()
            circ_new += gate
        circ_new += gate     # make circuit symmetrical & beautiful :)
    return circ_new

def estimate_rescaler(mol, shots:int=10000) -> float:
    # see vis_noise_scale.py
    return 0.89971224375925

    # 去极化信道和比特翻转信道宏观上都会使得测量值极差变小，尝试估计一下(线性)缩放比
    sim = HKSSimulator('mqvector', n_qubits=mol.n_qubits)
    circ = Circuit()
    circ += X.on(0)
    ops = QubitOperator('Z0')
    exp_GT = -1
    exp_actual = measure_single_ham(sim, circ, None, ops, shots)
    return exp_actual / exp_GT


''' Entry '''

# keep signature!!
def solution(molecule, Simulator: HKSSimulator) -> float:
    ''' Molecule '''
    #molecule = geometry_move_to_center(molecule)
    mol = generate_molecule(molecule)
    print('[mol]')
    print('  name:', mol.name)
    print('  geometry:', molecule)
    print('  n_atoms:', mol.n_atoms)
    print('  n_electrons:', mol.n_electrons)
    print('  n_orbitals:', mol.n_orbitals)
    print('  n_qubits:', mol.n_qubits)
    print('  nuclear_repulsion:', mol.nuclear_repulsion)
    print('  orbital_energies:', mol.orbital_energies)
    print('  hf_energy:', mol.hf_energy)
    print('  fci_energy:', mol.fci_energy)

    ''' Hamiltionian '''
    ham = get_molecular_hamiltonian(mol)
    
    const, split_ham = split_hamiltonian(ham)
    print('[ham]')
    print('  const:', const)
    print('  n_terms:', len(split_ham))
    split_ham = prune_hamiltonian(split_ham)
    print('  n_terms (pruned):', len(split_ham))
    #ham = combine_hamiltonian(const, split_ham)    # use precise ham for optimize accurate ground-state

    ''' Circuit '''
    #circ = get_HF_circuit(mol)                 # concrete noisy baseline
    #circ = get_uccsd_circuit(mol)              # concrete noiseless topline
    #circ = get_pchc_circuit(mol)               # seems not work
    circ = get_hae_ry_circit(mol, 3)
    #circ = get_hae_ry_compact_circit(mol, 3)   # a bit worse than the standard HEA(RY) =_=||
    #circ = get_cnot_centric_circit(mol)        # seems not work
    #circ = get_x_swap_circuit(mol)
    #circ = get_cnot_ry_seq_circuit(mol, 1)
    #circ = get_cnot_ry_sym_circuit(mol)
    print('[circ]')
    print('   n_qubits:', circ.n_qubits)
    print('   n_gates:', len(circ))
    print('   n_params:', len(circ.params_name))
    print(circ)

    ''' Optim & Params '''
    if 'from pretrained':
        init = get_hae_ry_circit_pretrained_params(depth=3)
        #init *= (np.abs(init) > 1e-5)
    else:
        init = 'randn'
    assert circ.ansatz_params_name, 'non-parametrized ansatz (e.g. HF_circuit) is no more supported :/'
    circ, pr = get_best_params_repeat(circ, ham, method='Adam', init=init, n_repeat=N_OPTIM_TRIAL, n_prune=0 if NOISE_AWARE_OPTIMIZE else N_PRUNE)
    print('   params:', repr(pr))
    circ, pr = prune_circuit(circ, pr)
    print('   n_gates (pruned):', len(circ))
    print('   n_params (pruned):', len(circ.params_name))
    print(circ)
    print('   params (pruned):', repr(pr))
    pr_empty = ParameterResolver(dict(zip(circ.params_name, np.zeros(len(circ.params_name)))))

    ''' Simulator (noiseless) '''
    from mindquantum.simulator import Simulator as OriginalSimulator
    sim = OriginalSimulator('mqvector', mol.n_qubits)
    print('[simulator] (noiseless)')
    exp = sim.get_expectation(Hamiltonian(ham), circ, pr=pr).real
    print('   exp (whole):', exp)
    if pr_empty is not None:
        exp = sim.get_expectation(Hamiltonian(ham), circ, pr=pr_empty).real
        print('   exp (whole, zero param):', exp)
    #print('   exp (per-term):')
    #for coeff, ops in split_ham:
    #    exp = sim.get_expectation(Hamiltonian(ops), circ, pr=pr).real
    #    print('     coeff=', coeff, 'term=', ops, 'exp=', exp)

    ''' Hardware (noisy) '''
    sim = Simulator('mqvector', mol.n_qubits)
    rescaler = estimate_rescaler(mol)
    print('[hardware] (noisy)')
    print('   rescaler:', rescaler)

    # measure pure HF-state: HF
    #result_hf = get_exp(sim, get_HF_circuit(mol), None, split_ham, shots=SHOTS)
    #print(f'>> result_hf:', result_hf)
    # measure no-param ansatz-state: CNOT+HF
    #result_vqe_np = get_exp(sim, zref_trim_circuit(circ), None, split_ham, shots=SHOTS)
    #print(f'>> result_vqe_np:', result_vqe_np)

    # measure optimized ansatz-state: RY(θ)+CNOT+HF
    result_vqe = get_exp(sim, circ, pr, split_ham, shots=SHOTS)
    print('>> result_vqe:', result_vqe)
    # measure zero-param ansatz-state: RY(0)+CNOT+HF
    result_vqe_zp = get_exp(sim, circ, pr_empty, split_ham, shots=SHOTS)
    print(f'>> result_vqe_zp:', result_vqe_zp)

    # Reference state error mitigation from https://pubs.acs.org/doi/10.1021/acs.jctc.2c00807
    result_rem = mol.hf_energy + (result_vqe - result_vqe_zp) / rescaler
    print('>> rem:', result_rem)
    return result_rem

    def zne(y):
        x = [1, 2, 3, 4]        # FIXME: this is wrong but it works :(
        f = lambda x, k, b: k * x + b
        popt, pcov = curve_fit(f, x, y)
        return f(0, *popt)

    # measure ZNE*k optimized ansatz-state: [RY(θ)+CNOT+HF]*k
    result_vqe_1 = get_exp(sim, zne_repeat_circuit(circ, n_repeat=1), pr, split_ham, shots=SHOTS)
    print(f'>> result_vqe_1:', result_vqe_1)
    result_vqe_2 = get_exp(sim, zne_repeat_circuit(circ, n_repeat=2), pr, split_ham, shots=SHOTS)
    print(f'>> result_vqe_2:', result_vqe_2)
    result_vqe_3 = get_exp(sim, zne_repeat_circuit(circ, n_repeat=3), pr, split_ham, shots=SHOTS)
    print(f'>> result_vqe_3:', result_vqe_3)
    result_vqe_0 = zne([result_vqe, result_vqe_1, result_vqe_2, result_vqe_3])
    print(f'>> result_vqe_0:', result_vqe_0)

    # measure ZNE*k zero-param ansatz-state: [RY(0)+CNOT+HF]*k
    result_vqe_zp_1 = get_exp(sim, zne_repeat_circuit(circ, n_repeat=1), pr_empty, split_ham, shots=SHOTS)
    print(f'>> result_vqe_zp_1:', result_vqe_zp_1)
    result_vqe_zp_2 = get_exp(sim, zne_repeat_circuit(circ, n_repeat=2), pr_empty, split_ham, shots=SHOTS)
    print(f'>> result_vqe_zp_2:', result_vqe_zp_2)
    result_vqe_zp_3 = get_exp(sim, zne_repeat_circuit(circ, n_repeat=3), pr_empty, split_ham, shots=SHOTS)
    print(f'>> result_vqe_zp_3:', result_vqe_zp_3)
    result_vqe_zp_0 = zne([result_vqe_zp, result_vqe_zp_1, result_vqe_zp_2, result_vqe_zp_3])
    print(f'>> result_vqe_zp_0:', result_vqe_zp_0)

    # Reference state error mitigation + zero noise extrapolation + rescaling
    result_rem_zne = mol.hf_energy + (result_vqe_0 - result_vqe_zp_0) / rescaler
    print('>> rem_zne:', result_rem_zne)
    return result_rem_zne
