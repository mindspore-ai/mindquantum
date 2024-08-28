import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
mol_path = script_path+'/mol.csv'
from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.core.circuit import Circuit

def split_hamiltonian(ham: QubitOperator):
    const = 0
    split_ham = []
    for i, j in ham.split():
        if j == 1:
            const = i.const.real
        else:
            split_ham.append([i.const.real, j])
    return const, split_ham

def reshape_ham(split_ham):
    coeff_observables = []
    for ham_idx, (coeff, ops) in enumerate(split_ham):

        one_observable=[]
        for qubit in list(ops.terms.keys())[0]:

            one_observable.append([qubit[0],qubit[1]])

        coeff_observables.append([coeff, one_observable])
    return coeff_observables

from mindquantum.core.operators import TimeEvolution
from mindquantum.algorithm.nisq import Transform
from mindquantum.core.circuit import UN
from mindquantum.core.gates import X
from mindquantum import uccsd_singlet_generator


def get_ucc_circ(mol):
    ucc = Transform(uccsd_singlet_generator(
        mol.n_qubits, mol.n_electrons)).jordan_wigner().imag
    ucc = TimeEvolution(ucc).circuit

    return UN(X, mol.n_electrons) + ucc

import numpy as np
from mindquantum.simulator import Simulator

from scipy.optimize import minimize
def get_best_params(mol, ham):
    circ = get_ucc_circ(mol)
    p0 = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
    grad_ops = Simulator('mqvector', circ.n_qubits).get_expectation_with_grad(
        Hamiltonian(ham),circ)
    

    def fun(x, grad_ops, energy_list=[]):
        f, g = grad_ops(x)
        f = f.real[0,0]
        g = g.real[0,0]

        return f, g
    
    res = minimize(fun, p0, (grad_ops, ), 'bfgs', True)
    
    return res.x, res.fun

def find_min_params(mol, ham):
    p_set = []
    f_set = []
    for k in range(10):
        p, f = get_best_params(mol, ham)
        p_set.append(p)
        f_set.append(f)

    f_final = np.min(f_set)
    f_index = f_set.index(f_final)
    p_final = p_set[f_index]

    return p_final, f_final

def rotate_to_z_axis_and_add_measure(circ, one_observable):
    circ = circ.copy()

    for qubit_position, oper in one_observable:
        

        if oper == 'X':
            circ.ry(-np.pi/2, qubit_position)
        elif oper == 'Y':
            circ.rx(np.pi/2, qubit_position)
        circ.measure(qubit_position)
    return circ 


def demo_full_measurement(coeff_observables, sim, shots = 100):
    full_ob_expec = 0
    for coeff, one_observable in coeff_observables:
        one_ob_expec = 0
        mea_circ = rotate_to_z_axis_and_add_measure(Circuit(), one_observable)
        one_ob_outcome = sim.sampling(mea_circ, shots=shots)
        for i, j in one_ob_outcome.data.items():
            one_ob_expec += (-1)**i.count('1')*j/shots
        full_ob_expec += one_ob_expec * coeff
    
    return full_ob_expec

def data_process(y, a, scaling=[1,2,3,4,5,6,7,8,9]):
    y = -np.array(y)
    def method_P(y,order = 1 ,scaling=scaling):
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
        return mitigated
    P_exp = -method_P(y,order=2)
    return P_exp

from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.algorithm.error_mitigation import fold_at_random
def solution(molecule, Simulator: HKSSimulator) -> float:

    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    coeff_observables = reshape_ham(split_ham)

    shots = 1000

    ucc = get_ucc_circ(mol)
    p, f = find_min_params(mol, ham)


    optimial_p = ParameterResolver(dict(zip(ucc.params_name, p)))
    

    sim = Simulator('mqvector', ucc.n_qubits)

    sim.apply_circuit(ucc, pr = optimial_p)




    final_ob_set = []
    for i in range(1,10):
        final_ucc = fold_at_random(ucc,i,'globally')
        sim = Simulator('mqvector', final_ucc.n_qubits)
        sim.apply_circuit(final_ucc, pr = optimial_p)

        k = int(shots/1000)
        ob_expec_set=[]
        for k_rep in range(k):
            full_ob_expec = demo_full_measurement(
                        coeff_observables, sim, shots = 1000)
            ob_expec_set.append(full_ob_expec)
        final_ob_expec = const + np.median(ob_expec_set) 
        final_ob_set.append(final_ob_expec)
    final_result = data_process(final_ob_set,a=0.39)



    return final_result


if __name__ == '__main__':
    import simulator
    simulator.init_shots_counter()
    molecule = read_mol_data(mol_path)
    sim = HKSSimulator
    result = solution(molecule, sim)
    print(sim, result)

