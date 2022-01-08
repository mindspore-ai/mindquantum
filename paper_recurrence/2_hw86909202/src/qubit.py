import os
os.environ['OMP_NUM_THREADS'] = '8'
import re
import sys
import random
import numpy as np
from openfermion.chem import MolecularData
from openfermion.ops.operators.qubit_operator import QubitOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner

from mindquantum import Circuit, X, RX, Hamiltonian, Simulator, generate_uccsd, AP
from mindquantum.core.operators import QubitOperator as hiq_operator
from mindspore import Tensor
from src.op_pool import generate_qccsd_pool
import mindspore.context as context
from scipy.optimize import minimize

# np.set_printoptions(threshold=np.inf)
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
TOLERANCE = 1e-8

def energy_obj(n_paras, energy, res_last={}):
    res_energy = energy(n_paras)
    f = np.real(res_energy[0])[0][0]
    grad = np.real(res_energy[1])[0][0]

    if 'f_last' not in res_last:
        res_last['f_last'] = 0.0
    else:
        res_last['f_last'] = res_last['f']
    
    res_last['x'] = list(n_paras)
    res_last['grad'] = list(grad)
    res_last['grad_norm'] = np.sqrt(np.sum(grad**2))
    res_last['f'] = f
    return f, grad


def qubit_adapt_vqe_a(molecule):
    molecule.load()
    print("hf:{}.".format(molecule.hf_energy))
    print("ccsd:{}.".format(molecule.ccsd_energy))
    print("fci:{}.".format(molecule.fci_energy))

    hamiltonian_InteractionOperator = molecule.get_molecular_hamiltonian()
    hamiltonian_FermionOperator = get_fermion_operator(hamiltonian_InteractionOperator)
    hamiltonian_QubitOperator = jordan_wigner(hamiltonian_FermionOperator)
    hamiltonian_QubitOperator.compress()
    for term in hamiltonian_QubitOperator.terms:
        hamiltonian_QubitOperator.terms[term] = hamiltonian_QubitOperator.terms[term].real
    # print(hamiltonian_QubitOperator)

    hf_circuit = Circuit([X.on(i) for i in range(0,molecule.n_electrons,1)])   # 1¦00001111⟩
    total_circuit = Circuit()
    total_circuit += hf_circuit
    operator_pools, circuit_pools = generate_qccsd_pool(molecule, th=-1)
    commuts = []
    for op_i in operator_pools:
        commut = hamiltonian_QubitOperator * QubitOperator(str(op_i)) - QubitOperator(str(op_i)) * hamiltonian_QubitOperator
        commut.compress()
        for term in commut.terms:
            commut.terms[term] = commut.terms[term].real
        commuts.append(commut)

    sim = Simulator('projectq', molecule.n_qubits)
    sim.apply_circuit(total_circuit)
    op_grad = np.zeros(len(commuts))
    for i, commut_i in enumerate(commuts):
        grad = sim.get_expectation(Hamiltonian(commut_i))
        op_grad[i] = abs(float(grad.real))
    op_grad_sort_idx = op_grad.argsort()[::-1]
    total_circuit += AP(circuit_pools[op_grad_sort_idx[0]], f'i{0}')
    init_amplitudes = np.array([random.random() for j in range(len(total_circuit.params_name))])

    final_res = 0.0
    max_iter_num = 100
    results = []
    for iter_i in range(max_iter_num):
        # print(total_circuit.summary())
        # print(total_circuit.parameter_resolver())

        sim.reset()
        energy = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_QubitOperator), total_circuit)
        res_last={}
        res = minimize(energy_obj,
            init_amplitudes,
            args=(energy, res_last),
            method='bfgs',
            options={'disp': False, 'return_all': True},
            jac=True,
            tol=1e-6)

        print("Step %3d energy %20.16f"%(iter_i, float(res.fun)))
        # print("Corresponding parameters:\n{}".format(res.x))
        results.append(abs(float(res.fun)-molecule.fci_energy))

        if abs(final_res - float(res.fun)) < TOLERANCE:
            print("Iterative is convergence!")
            print("Final energy : %20.16f"%(float(res.fun)))
            print("Final error : %20.16f"%(abs(molecule.fci_energy - float(res.fun))))
            break
        final_res = float(res.fun)

        sim.apply_circuit(total_circuit, res.x)
        for i, commut_i in enumerate(commuts):
            grad = sim.get_expectation(Hamiltonian(commut_i))
            op_grad[i] = abs(float(grad.real))

        op_grad_sort_idx = op_grad.argsort()[::-1]
        total_circuit += AP(circuit_pools[op_grad_sort_idx[0]], f'i{iter_i + 1}')
        init_amplitudes = np.zeros(len(total_circuit.params_name))
        init_amplitudes[:-1:1] = res.x
        init_amplitudes[-1] = random.random()
        
    return results


def qubit_adapt_vqe_b(molecule):
    molecule.load()
    print("hf:{}.".format(molecule.hf_energy))
    print("ccsd:{}.".format(molecule.ccsd_energy))
    print("fci:{}.".format(molecule.fci_energy))

    hamiltonian_InteractionOperator = molecule.get_molecular_hamiltonian()
    hamiltonian_FermionOperator = get_fermion_operator(hamiltonian_InteractionOperator)
    hamiltonian_QubitOperator = jordan_wigner(hamiltonian_FermionOperator)
    hamiltonian_QubitOperator.compress()
    for term in hamiltonian_QubitOperator.terms:
        hamiltonian_QubitOperator.terms[term] = hamiltonian_QubitOperator.terms[term].real
    # print(hamiltonian_QubitOperator)

    hf_circuit = Circuit([X.on(i) for i in range(0,molecule.n_electrons,1)])   # 1¦00001111⟩
    total_circuit = Circuit()
    total_circuit += hf_circuit
    operator_pools, circuit_pools = generate_qccsd_pool(molecule, th=-1)
    commuts = []
    for op_i in operator_pools:
        commut = hamiltonian_QubitOperator * QubitOperator(str(op_i)) - QubitOperator(str(op_i)) * hamiltonian_QubitOperator
        commut.compress()
        for term in commut.terms:
            commut.terms[term] = commut.terms[term].real
        commuts.append(commut)

    sim = Simulator('projectq', molecule.n_qubits)
    sim.apply_circuit(total_circuit)
    op_grad = np.zeros(len(commuts))
    for i, commut_i in enumerate(commuts):
        grad = sim.get_expectation(Hamiltonian(commut_i))
        op_grad[i] = abs(float(grad.real))
    op_grad_sort_idx = op_grad.argsort()[::-1]
    total_circuit += AP(circuit_pools[op_grad_sort_idx[0]], f'i{0}')
    init_amplitudes = np.array([random.random() for j in range(len(total_circuit.params_name))])

    final_res = 0.0
    max_iter_num = 100
    results = []
    for iter_i in range(max_iter_num):
        # print(total_circuit.summary())
        # print(total_circuit.parameter_resolver())
        # print(init_amplitudes)

        sim.reset()
        energy = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_QubitOperator), total_circuit)
        res_last={}
        res = minimize(energy_obj,
            init_amplitudes,
            args=(energy, res_last),
            method='bfgs',
            options={'disp': False, 'return_all': True},
            jac=True,
            tol=1e-6)

        print("Step %3d energy %20.16f"%(iter_i, float(res.fun)))
        # print("Corresponding parameters:\n{}".format(res.x))
        results.append(abs(float(res.fun)-molecule.fci_energy))

        if abs(final_res - float(res.fun)) < TOLERANCE:
            print("Iterative is convergence!")
            print("Final energy : %20.16f"%(float(res.fun)))
            print("Final error : %20.16f"%(abs(molecule.fci_energy - float(res.fun))))
            break
        final_res = float(res.fun)

        sim.apply_circuit(total_circuit, res.x)
        for i, commut_i in enumerate(commuts):
            grad = sim.get_expectation(Hamiltonian(commut_i))
            op_grad[i] = abs(float(grad.real))

        op_grad_sort_idx = op_grad.argsort()[::-1]
        total_circuit += AP(circuit_pools[op_grad_sort_idx[0]], f'i{iter_i + 1}')
        init_amplitudes = np.array([random.random() for j in range(len(total_circuit.params_name))])
        
    return results

