# Example program. (No meaningful result)
# The judging program "eval.py" will call `Main.run()` function from "src/main.py",
# and receive an energy value. So please put the executing part of your algorithm
# in `Main.run()`, return an energy value.
# All of your code should be placed in "src/" folder.

import numpy as np
import sys
import itertools
from mindquantum.core.gates import X, RY, RZ
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.algorithm.nisq import get_qubit_hamiltonian, HardwareEfficientAnsatz, generate_uccsd
from openfermionpyscf import run_pyscf
from openfermion.chem import MolecularData
from scipy.optimize import minimize


def simple_ccsd(circ, nq, ne, paras, max_dist=1):
    for (i,j) in itertools.combinations(range(nq),2):
        if abs(i-j)<=max_dist:# and i%2==j%2: (full exchange vs. spin-reserved exchange)
            circ += X.on(i,j)
            circ += RY(f"{paras}_{i}_{j}").on(j,i)
            circ += X.on(i,j)

def excited_state_solver(mol):
    uccsd_ansatz, x0, _, ham_op, n_qubits, n_electrons = generate_uccsd(mol)
    hartreefock_wfn_circuit = Circuit([X.on(i) for i in range(n_electrons)])
    gs_circ = hartreefock_wfn_circuit + uccsd_ansatz

    gs_sim = Simulator("mqvector", n_qubits)
    grad_ops = gs_sim.get_expectation_with_grad(Hamiltonian(ham_op), gs_circ)

    def func(x, grad_ops):
        f, g = grad_ops(x)
        return np.real(np.squeeze(f)), np.real(np.squeeze(g))

    gs_res = minimize(func, x0, args=(grad_ops), method="bfgs", jac=True)

    gs_pr = dict(zip(gs_circ.params_name, gs_res.x))

    gs_sim.apply_circuit(gs_circ, gs_pr)

    gs_en = gs_sim.get_expectation(Hamiltonian(ham_op)).real

    print("Ground state energy: ", gs_en)

    def es_func(x,grad_ops,ov_ops,beta):
        f0, g0 = grad_ops(x)
        f1, g1 = ov_ops(x)
        f0, g0, f1, g1 = (np.squeeze(f0), np.squeeze(g0), np.squeeze(f1), np.squeeze(g1))
        cost = np.real(f0) + beta * np.abs(f1) ** 2  
        punish_g = np.conj(g1) * f1 + g1 * np.conj(f1) 
        total_g = g0 + beta * punish_g
        print(np.real(f0))
        return cost, total_g.real

    es_circ = hartreefock_wfn_circuit

    # number of layers 
    simple_ccsd(es_circ,n_qubits,n_electrons,"g0",n_qubits-1)
    #simple_ccsd(es_circ,n_qubits,n_electrons,"g1",n_qubits-1)
    #simple_ccsd(es_circ,n_qubits,n_electrons,"g2",n_qubits-1)

    '''
    for i in range(n_qubits):
        for j in range(i):
            es_circ += X.on(i,j)
            es_circ += RY("basic_{}_{}".format(i,j)).on(j,i)
            es_circ += X.on(i,j)
    '''
            
    es_sim = Simulator("mqvector", mol.n_qubits)

    beta = 1.0

    es_grad_ops = es_sim.get_expectation_with_grad(Hamiltonian(ham_op), es_circ)

    ip_grad_ops = es_sim.get_expectation_with_grad(
        Hamiltonian(QubitOperator("")), es_circ, Circuit(), simulator_left=gs_sim
    )

    coup_grad_ops = es_sim.get_expectation_with_grad(
        Hamiltonian(ham_op), es_circ, Circuit(), simulator_left=gs_sim
    )

    init_amplitudes = np.random.random(len(es_circ.all_paras))

    es_res = minimize(
        es_func,
        init_amplitudes,
        args=(es_grad_ops, ip_grad_ops, beta),
        method="bfgs",
        jac=True,
    )

    es_pr = dict(zip(es_circ.params_name, es_res.x))

    es_sim.apply_circuit(es_circ, es_pr)

    es_en = es_sim.get_expectation(Hamiltonian(ham_op)).real  
    
    print("First excited state energy: ", es_en)

    return es_en


class Main:
    def run(self, mol):
        return excited_state_solver(mol)
