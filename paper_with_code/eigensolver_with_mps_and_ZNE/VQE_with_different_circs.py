import os
import sys

sys.path.append(os.path.abspath(__file__))
from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian, TimeEvolution
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import X, RY, RZ, RX, CNOT
from mindquantum import uccsd_singlet_generator, SingleLoopProgress
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.algorithm.nisq import uccsd0_singlet_generator, HardwareEfficientAnsatz, QubitUCCAnsatz, StronglyEntangling

import numpy as np
from scipy.optimize import minimize
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import Transform


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


def get_ucc_circ(mol, Ansatz='UCCSD'):
    if Ansatz == 'UCCSD':
        ucc = Transform(uccsd_singlet_generator(
            mol.n_qubits, mol.n_electrons)).jordan_wigner().imag
        ucc = TimeEvolution(ucc).circuit
        print("uccsd_singlet_generator")
    elif Ansatz == 'UCCSD0':
        ucc = Transform(uccsd0_singlet_generator(
            mol.n_qubits, mol.n_electrons)).jordan_wigner().imag
        ucc = TimeEvolution(ucc).circuit
        print("uccsd_singlet_generator")
    elif Ansatz == 'HardwareEfficient':
        ucc = HardwareEfficientAnsatz(mol.n_qubits, [RX, RY, RZ]).circuit
        print("HardwareEfficientAnsatz")
    elif Ansatz == 'QubitUCCAnsatz':
        ucc = QubitUCCAnsatz(mol.n_qubits, mol.n_electrons, trotter_step=2).circuit
        print("QubitUCCAnsatz")
    elif Ansatz == 'StronglyEntangling':
        ucc = StronglyEntangling(mol.n_qubits, depth=4, entangle_gate=X).circuit
        print("StronglyEntangling")
    else:
        raise ValueError('Ansatz not found')

    ucc.summary()
    return UN(X, mol.n_electrons) + ucc


def get_best_params(mol, ham, Ansatz='UCCSD'):
    circ = get_ucc_circ(mol, Ansatz=Ansatz)
    p0 = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
    grad_ops = Simulator('mqvector', circ.n_qubits).get_expectation_with_grad(
        Hamiltonian(ham), circ)

    def fun(x, grad_ops):
        f, g = grad_ops(x)
        f = f.real[0, 0]
        g = g.real[0, 0]
        return f, g

    res = minimize(fun, p0, (grad_ops, ), 'bfgs', True)
    return res.x


def mea_single_ham(circ, ops, p, Simulator: HKSSimulator, shots=100):
    circ = rotate_to_z_axis_and_add_measure(circ, ops)
    pr = ParameterResolver(dict(zip(circ.params_name, p)))
    sim = Simulator('mqvector', circ.n_qubits)
    result = sim.sampling(circ, shots=shots, pr=pr)
    expec = 0
    for i, j in result.data.items():
        expec += (-1)**i.count('1') * j / shots
    return expec


def solution(molecule, Simulator: HKSSimulator, Ansatz='UCCSD') -> float:
    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    # print('ham:', ham)
    const, split_ham = split_hamiltonian(ham)
    ucc = get_ucc_circ(mol, Ansatz=Ansatz)
    p = get_best_params(mol, ham, Ansatz=Ansatz)
    result = const
    with SingleLoopProgress(len(split_ham), '哈密顿量测量中') as bar:
        for idx, (coeff, ops) in enumerate(split_ham):
            result += mea_single_ham(ucc, ops, p, Simulator) * coeff
            bar.update_loop(idx)
    return result


if __name__ == '__main__':
    import simulator
    simulator.init_shots_counter()


    for data_path, mol_name in [('data_mol/mol_H4.csv', 'H4'),('data_mol/mol_H2O.csv', 'H2O'),('data_mol/mol_LiH.csv', 'LiH')]:# ('data_mol/mol_H4.csv', 'H4'),('data_mol/mol_H2O.csv', 'H2O'),('data_mol/mol_LiH.csv', 'LiH')
        # data_path = 'mol_test.csv'
        molecule = read_mol_data(data_path)
        print(f"mol_name: {mol_name}")
        for Ansatz in ['HardwareEfficient', 'UCCSD0', 'UCCSD', 'QubitUCCAnsatz', 'StronglyEntangling']:  #
            print(f"Ansatz: {Ansatz}")
            HKSSimulator_energy = []
            Simulator_energy = []
            for num in range(1):
                for sim in [HKSSimulator, Simulator]:
                # for sim in [Simulator]:
                    result = solution(molecule, sim, Ansatz=Ansatz)
                    print(sim, result)
                    if sim == HKSSimulator:
                        HKSSimulator_energy.append(result)
                    else:
                        Simulator_energy.append(result)
                print(f"finished {num}th for {Ansatz}")

            # save_path = 'Qubit_UCC_results'
            # save_path = 'HardwareEfficient_results'
            # save_path = 'UCCSD_results'
            # save_path = 'StronglyEntangling_results'
            # save_path = 'results/' + mol_name + '/' + Ansatz + '_results'
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # np.save(f'{save_path}/HKSSimulator_energy.npy', HKSSimulator_energy)
            # np.save(f'{save_path}/Simulator_energy.npy', Simulator_energy)
