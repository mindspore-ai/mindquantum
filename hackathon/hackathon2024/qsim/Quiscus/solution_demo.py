import os, sys
sys.path.append(os.path.abspath(__file__))

import numpy as np
from scipy.optimize import minimize
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.algorithm.nisq import Transform
from mindquantum.utils.progress import SingleLoopProgress
from mindquantum.third_party.unitary_cc import uccsd_singlet_generator

from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data


def split_hamiltonian(ham: QubitOperator):
    const = 0
    split_ham = []
    for i, j in ham.split():
        if j == 1:
            const = i.const.real
        else:
            split_ham.append([i.const.real, j])
    return const, split_ham


def rotate_to_z_axis_and_add_measure(circ: Circuit, ops: QubitOperator):
    circ = circ.copy()
    assert ops.is_singlet
    for idx, o in list(ops.terms.keys())[0]:
        if o == 'X':
            circ.ry(-np.pi / 2, idx)
        elif o == 'Y':
            circ.rx(np.pi / 2, idx)
        circ.measure(idx)
    return circ


def get_ucc_circ(mol):
    ucc = Transform(uccsd_singlet_generator(mol.n_qubits, mol.n_electrons)).jordan_wigner().imag
    ucc = TimeEvolution(ucc).circuit
    return UN(X, mol.n_electrons) + ucc


def get_best_params(mol, ham):
    circ = get_ucc_circ(mol)
    p0 = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
    grad_ops = Simulator('mqvector', circ.n_qubits).get_expectation_with_grad(Hamiltonian(ham), circ)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    res = minimize(fun, p0, (grad_ops, ), 'bfgs', True)
    return res.x


def mea_single_ham(circ, ops, p, Simulator: HKSSimulator, shots=100) -> float:
    circ = rotate_to_z_axis_and_add_measure(circ, ops)
    pr = ParameterResolver(dict(zip(circ.params_name, p)))
    sim = Simulator('mqvector', circ.n_qubits)
    result = sim.sampling(circ, shots=shots, pr=pr)
    expec = 0
    for i, j in result.data.items():
        expec += (-1)**i.count('1') * j / shots
    return expec


def solution(molecule, Simulator: HKSSimulator) -> float:
    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    ucc = get_ucc_circ(mol)
    p = get_best_params(mol, ham)
    result = const
    with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
        for idx, (coeff, ops) in enumerate(split_ham):
            result += mea_single_ham(ucc, ops, p, Simulator) * coeff
            bar.update_loop(idx)
    return result


if __name__ == '__main__':
    import simulator
    simulator.init_shots_counter()
    molecule = read_mol_data('mol.csv')
    for sim in [HKSSimulator, Simulator]:
        result = solution(molecule, sim)
        print(sim, result)
