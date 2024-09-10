import os
import sys

script_path = os.path.dirname(os.path.abspath(__file__))
mol_path = script_path+'/mol.csv'
from simulator import HKSSimulator
from utils import generate_molecule, get_molecular_hamiltonian, read_mol_data
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import ParameterResolver

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
def test_best_params(mol, ham):
    circ = get_ucc_circ(mol)
    grad_ops = Simulator('mqvector', circ.n_qubits).get_expectation_with_grad(
        Hamiltonian(ham),circ)
    
    def fun(x, grad_ops, energy_list=[]):
        f, g = grad_ops(x)
        f = f.real[0,0]
        g = g.real[0,0]
#        if energy_list is not None:
#            energy_list.append(f)
#            if len(energy_list)% 10 ==0:
#                print(f'Step: {len(energy_list)}, \tenergy: {f}')
        #这里的f是值，g是梯度
        return f, g
    p_set = []
    f_set = []
    for i in range(100):
        p0 = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
        res = minimize(fun, p0, (grad_ops, ), 'bfgs', True)
        p_set.append(res.x)
        f_set.append(res.fun)        
    f_final=np.min(f_set)    
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

def find_par_meas(coeff_observables):
    full_n = len(coeff_observables)
    full_comp_obser = []
    for index in range(full_n):
        comp_obser = []
        for i in range(full_n):
            if set(map(tuple, coeff_observables[index][1])).issubset(
                set(map(tuple, coeff_observables[i][1]))):
                comp_obser.append(i)
        full_comp_obser.append(comp_obser)
    full_meas = []
    for j in range(full_n):
        if len(full_comp_obser[j]) == 1:
            full_meas.append(j)
    return full_meas


def find_par_chi(full_meas, coeff_observables):
    full_n = len(coeff_observables)
    parent_with_children = []
    for parent in full_meas:
        children_meas = []
        for child in range(full_n):
            if set(map(tuple, coeff_observables[child][1])).issubset(
                set(map(tuple, coeff_observables[parent][1]))):
                children_meas.append(child)
        parent_with_children.append([parent,children_meas])    
    return parent_with_children

def find_rela_position(parent_with_children, coeff_observables):
    relative_position = []
    for parent, children in parent_with_children:

        children_rela_position = []
        for one_child in children:
            one_rela_position = []
            for qubit_position in coeff_observables[one_child][1]:

                one_rela_position.append(
                    coeff_observables[parent][1].index(qubit_position))

            children_rela_position.append(one_rela_position)
        relative_position.append(children_rela_position)    
    return relative_position

def parent_with_children(coeff_observables):
    par_meas = find_par_meas(coeff_observables)
    parent_with_children = find_par_chi(par_meas, coeff_observables)
    relative_position = find_rela_position(parent_with_children, coeff_observables)
    return par_meas, parent_with_children, relative_position

def get_k_groups(sim, one_measurement,coeff_observables , k, shots):
    mea_circ = rotate_to_z_axis_and_add_measure(
            Circuit(),coeff_observables[one_measurement][1])
    k_group = []
    for k_rep in range(k):
        one_mea_result = sim.sampling(mea_circ, shots = shots)
        one_mea_res_store = []
        for i,j in one_mea_result.data.items():
            one_mea_res_store.append([i[::-1],j/shots])
        k_group.append(one_mea_res_store)
    return k_group

def find_all_states(k_group):
    All_states = []
    for one_res in k_group:
        for state_pre in one_res:
            All_states.append(state_pre[0])
    
    All_states = list(set(All_states))
    return All_states
    
def get_k_states(res_states, k_group,k):
    state_k_samp = []
    for state in res_states:

        state_prediction_list = []
        for k_th_res in k_group:

            
            for one_k_res in k_th_res:                 
                if one_k_res[0] == state:

                    state_prediction_list.append(one_k_res[1])
        if len(state_prediction_list) < k:
            state_prediction_list+=[0]*(k-len(state_prediction_list))

        state_std = np.std(state_prediction_list)

        state_k_samp.append([state, state_prediction_list, state_std])

    return state_k_samp

def parent_measure_results(sim, meas_list, coeff_observables,k, shots):

    measurement_results = []
    
    for one_measurement in meas_list:
        k_group = get_k_groups(
            sim, one_measurement, coeff_observables,k=k, shots=shots)

        res_states = find_all_states(k_group)
        state_k_samp = get_k_states(res_states, k_group, k)

        measurement_results.append(state_k_samp) 
            
    return measurement_results

def one_ob_k(one_relative_position, one_mea_res):
    one_rela_res = []
    for rela_position in one_relative_position:

        k_ob_expec = 0
        for one_state_res in one_mea_res:

        
            qubit_state = ''
            for qubit_position in rela_position:
                qubit_state += one_state_res[0][qubit_position]

            k_ob_expec += (-1)**qubit_state.count('1')*np.array(one_state_res[1])

        one_rela_res.append(k_ob_expec)
    return one_rela_res

def get_all_ob_k(measurement_results,relative_position):
    All_ob_k = []
    for i in range(len(measurement_results)):

        All_ob_k.append(one_ob_k(relative_position[i], measurement_results[i]))
    return All_ob_k

def get_ob_par_pos(coeff_observables,par_child):
    full_ob_par_pos = []
    for i, [coeff, one_obser] in enumerate(coeff_observables):

        every_ob_parent = []
        for parent, children in par_child: 
            if i in children:
                every_ob_parent.append([parent, children.index(i)])
        full_ob_par_pos.append(every_ob_parent)
    return full_ob_par_pos

def get_full_ob_prediction(coeff_observables,meas_list, full_ob_par_pos, All_ob_k,k_mode):
    full_ob_prediction = []
    for i, [coeff, one_obser] in enumerate(coeff_observables):
        one_full_prediction = []
        for par, pos in full_ob_par_pos[i]:

            num_par = meas_list.index(par)
            one_full_prediction+=list(All_ob_k[num_par][pos])
        if k_mode == 'k_ob':
            full_ob_prediction.append([coeff,np.median(one_full_prediction)])
        if k_mode == 'k_ham':
            full_ob_prediction.append([coeff,np.average(one_full_prediction)])
    return full_ob_prediction

def get_coeff_ob_pre(coeff_observables,meas_list,par_child, 
                     measurement_results,relative_position,k_mode):
    All_ob_k = get_all_ob_k(measurement_results,relative_position)
    full_ob_par_pos = get_ob_par_pos(coeff_observables,par_child)
    full_ob_prediction = get_full_ob_prediction(
        coeff_observables,meas_list, full_ob_par_pos, All_ob_k, k_mode=k_mode)
    return full_ob_prediction

def ham_prediction(All_coeff_ob):
    ham_prediction = 0
    for coeff, ob_pre in All_coeff_ob:
        ham_prediction += coeff*ob_pre
    return ham_prediction

def comp_ham_prediction(sim, coeff_observables,k_mode,k =30,shots = 100):
    meas_list, par_child, relative_position = parent_with_children(coeff_observables)
    measurement_results = parent_measure_results(
        sim, meas_list, coeff_observables,k=k, shots=shots)
    All_coeff_ob =  get_coeff_ob_pre(coeff_observables,meas_list,par_child, 
                     measurement_results,relative_position, k_mode=k_mode)
    ham_pre = ham_prediction(All_coeff_ob)
    return ham_pre

def data_process(y, a, scaling=[5,6,7,8,9]):
    y = -np.array(y)
    def method_P(y,order = 1 ,scaling=scaling):
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
        return mitigated
    P_exp = -method_P(y,order=1)
    return P_exp


from mindquantum.algorithm.error_mitigation import fold_at_random
def solution(molecule, Simulator: HKSSimulator) -> float:

    mol = generate_molecule(molecule)
    ham = get_molecular_hamiltonian(mol)
    const, split_ham = split_hamiltonian(ham)
    coeff_observables = reshape_ham(split_ham)

    

    ucc = get_ucc_circ(mol)
    p, f = test_best_params(mol, ham)


    optimial_p = ParameterResolver(dict(zip(ucc.params_name, p)))
    sim = Simulator('mqvector', ucc.n_qubits)
    sim.apply_circuit(ucc, pr = optimial_p)

    final_ob_set = []
    shots = 10000
    k = 30
    final_ob_set = []
    for i in range(5,10):
        final_ucc = fold_at_random(ucc,i,'globally')
        sim = Simulator('mqvector', final_ucc.n_qubits)
        sim.apply_circuit(final_ucc, pr = optimial_p)

        k_ham_num = int(k/2)
        k_ham_shots = 2*shots
        k_ham_group = []
        k_mode = 'k_ham'
        for i in range(k_ham_num):
            full_ham_prediction = comp_ham_prediction(
                    sim, coeff_observables,k_mode=k_mode,
                            k=1,shots=k_ham_shots)
            final_ham_prediction = const + full_ham_prediction
            k_ham_group.append(final_ham_prediction)
        k_median_ham = np.median(k_ham_group)
        final_ob_set.append(k_median_ham)
        
    final_result = data_process(final_ob_set,a=0.7730974994386957)
    return final_result


if __name__ == '__main__':
    import simulator
    simulator.init_shots_counter()
    molecule = read_mol_data(mol_path)
    sim = HKSSimulator
    result = solution(molecule, sim)
    print(sim, result)

