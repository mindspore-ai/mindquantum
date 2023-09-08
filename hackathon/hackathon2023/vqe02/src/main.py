# Example program. (No meaningful result)
# The judging program "eval.py" will call `Main.run()` function from "src/main.py",
# and receive an energy value. So please put the executing part of your algorithm
# in `Main.run()`, return an energy value.
# All of your code should be placed in "src/" folder.

import numpy as np
from mindquantum.algorithm.nisq import generate_uccsd
from scipy.optimize import minimize
import sys
from mindquantum.core.gates import X, RY, RZ
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.algorithm.nisq import get_qubit_hamiltonian, HardwareEfficientAnsatz
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData


def excited_state_solver(mol):
#     Get hamiltonian of the molecule
    ham_op = get_qubit_hamiltonian(mol)
    # print('对接成功')
    # print('ham_op = ', ham_op)# 没问题
    # Construct hartreefock wave function circuit
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(mol.n_electrons)])

    # Construct ground state ansatz circuit
    gs_circ = (
        hartreefock_wfn_circuit
        + HardwareEfficientAnsatz(mol.n_qubits, [RY, RZ], depth=10).circuit
    )

    # Declare the ground state simulator
    gs_sim = Simulator("mqvector", gs_circ.n_qubits)

    # Get the expectation and gradient calculating function
    gs_grad_ops = gs_sim.get_expectation_with_grad(Hamiltonian(ham_op), gs_circ)


    # Define the objective function to be minimized
    def gs_func(x, grad_ops):
        f, g = grad_ops(x)
        # print(np.real(np.squeeze(f)), np.real(np.squeeze(g)))
        return np.real(np.squeeze(f)), np.real(np.squeeze(g))


    # Initialize amplitudes
    init_amplitudes = np.random.random(len(gs_circ.all_paras))
    # print('gs_init_amplitudes = ', init_amplitudes)
    # print(len(gs_circ.all_paras))
    # init_amplitudes =  



    # Get Optimized result
    gs_res = minimize(gs_func, init_amplitudes, args=(gs_grad_ops), method="bfgs", jac=True,) 
       # gs_res = minimize(gs_func, init_amplitudes, args=(gs_grad_ops), method="bfgs", tol=1e-10, jac=True, options={'gtol': 1e-10, 'maxiter': 1e10}) 
    
    # Construct parameter resolver of the ground state circuit
    gs_pr = dict(zip(gs_circ.params_name, gs_res.x))

    # Evolve into ground state
    gs_sim.apply_circuit(gs_circ, gs_pr)

    # Calculate energy of ground state
    gs_en = gs_sim.get_expectation(Hamiltonian(ham_op)).real

    # -----------------------------------------------------------------

    # Construct excited state ansatz circuit
    es_circ = (
        hartreefock_wfn_circuit
        + HardwareEfficientAnsatz(mol.n_qubits, [RY, RZ], depth=10).circuit
    )

    # Declare the excited state simulator
    es_sim = Simulator("mqvector", mol.n_qubits)

    # Get the expectation and gradient calculating function
    es_grad_ops = es_sim.get_expectation_with_grad(Hamiltonian(ham_op), es_circ)

    # Get the expectation and gradient calculating function of inner product
    ip_grad_ops = es_sim.get_expectation_with_grad(
        Hamiltonian(QubitOperator("")), es_circ, Circuit(), simulator_left=gs_sim
    )

    # Define punishment coefficient
    beta = 100000
    # print('beta=29999')


    # Define the objective function to be minimized
    def es_func(x, es_grad_ops, ip_grad_ops, beta):
        f0, g0 = es_grad_ops(x)
        f1, g1 = ip_grad_ops(x)
        # Remove extra dimension of the array
        f0, g0, f1, g1 = (np.squeeze(f0), np.squeeze(g0), np.squeeze(f1), np.squeeze(g1))
        cost = np.real(f0) + beta * np.abs(f1) ** 2  # Calculate cost function
        punish_g = np.conj(g1) * f1 + g1 * np.conj(f1)  # Calculate punishment term gradient
        total_g = g0 + beta * punish_g
        return cost, total_g.real


    # Initialize amplitudes
    init_amplitudes = np.random.random(len(es_circ.all_paras))
    # print('es_init_amplitudes = ', init_amplitudes) 
    
    # print(len(es_circ.all_paras))
    # init_amplitudes = 

    

    # Get Optimized result
    es_res = minimize(
        es_func,
        init_amplitudes,
        args=(es_grad_ops, ip_grad_ops, beta),
        method="bfgs",jac=True,
    )
    # es = minimize(
    #     es_func,
    #     init_amplitudes,
    #     args=(es_grad_ops, ip_grad_ops, beta),
    #     method="bfgs",
    #      tol=1e-10, jac=True, options={'gtol': 1e-10, 'maxiter': 1e10}
    # )

    # Construct parameter resolver of the excited state circuit
    es_pr = dict(zip(es_circ.params_name, es_res.x))

    # Evolve into excited state
    es_sim.apply_circuit(es_circ, es_pr)

    # Calculate energy of excited state
    es_en = es_sim.get_expectation(Hamiltonian(ham_op)).real
    # print('主程跑完es_en = ' , es_en)
    return es_en

class Main:
    def run(self, mol):
        basis = "6-31g"
        # print('更新')
        return excited_state_solver(run_pyscf(
               MolecularData(mol.geometry, mol.basis, mol.multiplicity,
                             data_directory='./'), run_fci=1,))
