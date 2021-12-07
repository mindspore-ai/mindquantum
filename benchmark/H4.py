import os

os.environ['OMP_NUM_THREADS'] = '4'
import time
import json
import numpy as np
from numpy.core.arrayprint import _extendLine
from scipy.optimize import minimize

from mindquantum.algorithm.nisq.chem import HardwareEfficientAnsatz
from mindquantum.core import Hamiltonian, RY, RZ, X
from mindquantum import Simulator
from q_ham import q_ham_producer
from data import initdata
import mole


def energy_obj(n_paras, mol_pqc):
    global count
    count += 1

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


def HeaVqe(n_qubits, sparsed_q_ham, single_rot_gate_seq, entangle_gate,
           entangle_mapping, depth):
    hea = HardwareEfficientAnsatz(n_qubits, single_rot_gate_seq, entangle_gate,
                                  entangle_mapping, depth)
    hea_circuit = hea.circuit

    sim = Simulator('projectq', n_qubits)
    mol_pqc = sim.get_expectation_with_grad(sparsed_q_ham, hea_circuit)

    #para_only_energy_obj = partial(energy_obj, hea_circuit, q_ham)

    n_paras = len(hea_circuit.params_name)
    paras = [np.random.uniform(-np.pi, np.pi) for i in range(n_paras)]

    res = minimize(energy_obj,
                   paras,
                   args=(mol_pqc, ),
                   method='bfgs',
                   jac=True,
                   tol=1e-6)
    #print("VQE energy with HEA ansatz:{}".format(float(res.fun)))
    # print("Corresponding parameters:{}".format(res.x.tolist()))

    return n_paras, float(res.fun)


if __name__ == "__main__":
    mole_name = 'H4'
    basis = 'sto3g'
    charge = 0
    multiplicity = 1
    transform = 'jordan_wigner'

    single_rot_gate_seq, entangle_gate, \
    entangle_mapping = [RY, RZ], X, 'linear'

    # Only run the next line at the first time for each molecule
    initdata(mole_name)

    with open(os.path.join(
            os.getcwd(),
            r'data/mindquantum_energies_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        energies = json.loads(data)

    with open(os.path.join(
            os.getcwd(), r'data/mindquantum_times_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        times = json.loads(data)

    with open(os.path.join(
            os.getcwd(),
            r'data/mindquantum_parameters_{}.json'.format(mole_name)),
              'r+',
              newline='') as f:
        data = f.read()
        parameters = json.loads(data)

    bond_lengths = [0.1 * i + 0.4 for i in range(17)]
    bbond_lengths = [0.2 * i + 2.2 for i in range(5)]
    bond_lengths.extend(bbond_lengths)

    for bond_len in bond_lengths:
        # need to change
        geometry = getattr(mole, 'get_{}_geo'.format(mole_name))(bond_len)
        n_qubits,n_electrons, \
        hf_energy, ccsd_energy, \
        fci_energy, q_ham = q_ham_producer(geometry, basis, charge, multiplicity, transform)
        sparsed_q_ham = Hamiltonian(q_ham)
        sparsed_q_ham.sparse(n_qubits)

        count = 0
        depth = 1
        # start time
        start = time.time()
        while (count < 5e4):
            energy = 0
            for i in range(10):
                (n_paras, e_vqe) = HeaVqe(n_qubits, sparsed_q_ham,
                                          single_rot_gate_seq, entangle_gate,
                                          entangle_mapping, depth)
                energy = e_vqe
                print(bond_len, count, n_paras, energy, fci_energy)
                """ if abs(e_vqe-fci_energy) < 1.6e-2:
                    break
                energy += e_vqe/10 """
                if abs(energy - fci_energy) < 1.6e-3:
                    break
            depth += 1

            if abs(energy - fci_energy) < 1.6e-3:
                break
        # time cost
        t_cost = time.time() - start
        energies["bond_lengths"].append(bond_len)
        energies["energies"]["Hartree-Fock"].append(hf_energy)
        energies["energies"]["full-CI"].append(fci_energy)
        energies["energies"]["CCSD"].append(ccsd_energy)
        energies["energies"]["HEA"].append(energy)
        times["bond_lengths"].append(bond_len)
        times["times"]["HEA"].append(t_cost)
        parameters["bond_lengths"].append(bond_len)
        parameters["parameters"]["HEA"].append(n_paras)

    with open(os.path.join(
            os.getcwd(),
            r'data/mindquantum_energies_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(energies)
        f.write(b)

    with open(os.path.join(
            os.getcwd(), r'data/mindquantum_times_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(times)
        f.write(b)

    with open(os.path.join(
            os.getcwd(),
            r'data/mindquantum_parameters_{}.json'.format(mole_name)),
              'w+',
              newline='') as f:
        b = json.dumps(parameters)
        f.write(b)