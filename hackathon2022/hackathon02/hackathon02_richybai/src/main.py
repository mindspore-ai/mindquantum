import os
from pickletools import optimize
from tabnanny import verbose
os.environ['OMP_NUM_THREADS'] = '1'
import sys
from hiqfermion.drivers import MolecularData
# from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
import time
import numpy as np
from mindquantum import *
from scipy.optimize import minimize
from mindquantum.algorithm import get_qubit_hamiltonian
from mindquantum.algorithm.nisq.chem import QubitUCCAnsatz
from mindquantum.core.parameterresolver import ParameterResolver as PR
import matplotlib.pyplot as plt
import mindspore as ms
import itertools
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def _single_qubit_excitation_circuit(i, k, theta):
    """
    Implement circuit for single qubit excitation.
    k: creation
    """
    circuit_singles = Circuit()
    circuit_singles += CNOT(i, k)
    circuit_singles += RY(theta).on(k, i)
    circuit_singles += CNOT(i, k)
    return circuit_singles


def _double_qubit_excitation_circuit(i, j, k, l, theta):
    """
    Implement circuit for double qubit excitation.
    k, l: creation
    """
    circuit_doubles = Circuit()
    circuit_doubles += CNOT.on(k, l)
    circuit_doubles += CNOT.on(i, j)
    circuit_doubles += CNOT.on(j, l)
    circuit_doubles += X.on(k)
    circuit_doubles += X.on(i)
    circuit_doubles += RY(theta).on(l, ctrl_qubits=[i, j, k])
    circuit_doubles += X.on(k)
    circuit_doubles += X.on(i)
    circuit_doubles += CNOT.on(j, l)
    circuit_doubles += CNOT.on(i, j)
    circuit_doubles += CNOT.on(k, l)
    return circuit_doubles


class Timer:
    def __init__(self, t0=0.0):
        self.start_time = time.time()
        self.t0 = t0

    def runtime(self):
        return time.time() - self.start_time + self.t0


def format_time(t):
    hh = t // 3600
    mm = (t - 3600 * hh) // 60
    ss = t % 60
    return hh, mm, ss


def func(x, grad_ops, file, target, show_iter_val=False):
    f, g = grad_ops(x)
    if show_iter_val:
        print(np.squeeze(f), file=file)
        sys.stdout.flush()
    if np.squeeze(f) - target < 0.0016:
        return np.real(np.squeeze(f)), np.zeros_like(g) # 直接把梯度输出为0,结束运行
    else:
        return np.real(np.squeeze(f)), np.squeeze(g)
    

def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


def build_encoder(num_bit, num_electrons):
    encoder = Circuit()
    for i in range(num_electrons):
        encoder += X.on(i)
        # encoder += H.on(i)
    return encoder


def build_ansatz(num_bit, num_electrons):
    ansatz = Circuit()

    QUCC = QubitUCCAnsatz(num_bit, num_electrons, trotter_step=1)
    ansatz += QUCC.circuit
    # 在这里拓展线路争取把 CH4 0.8 算出来
    # count = 0
    # circuit = Circuit()
    # for i, j in itertools.product(range(num_electrons), range(num_electrons, num_bit)):
    #     theta = PR({f'{count}': 1})
    #     count += 1
    #     ansatz += _single_qubit_excitation_circuit(i, j, theta)
    
    for i in range(num_bit):
        if i+1 < num_bit:
            ansatz += H.on(i, i+1)
            ansatz += RX(f"rx-{i}").on(i, i+1)
            ansatz += H.on(i, i+1)
            ansatz += H.on(i, i+1)
            ansatz += RY(f"ry-{i}").on(i, i+1)
            ansatz += H.on(i, i+1)
            ansatz += H.on(i, i+1)
            ansatz += RZ(f"rx-{i}").on(i, i+1)
            ansatz += H.on(i, i+1)
    
    # for i, j in itertools.product(range(num_electrons), range(num_electrons)):
    #     if i >= j:
    #         continue
    #     for k, l in itertools.product(range(num_electrons, num_bit), range(num_electrons, num_bit)):
    #         if k >= l:
    #             continue
    #         theta = PR({f'{count}': 1})
    #         count += 1
    #         ansatz += _double_qubit_excitation_circuit(i, j, k, l, theta)
    
    # print(num_bit, num_electrons)
    # ansatz.summary()
    return ansatz


class VQEoptimizer:


    def __init__(self, seed=1202, file=None):
        self.timer = Timer()
        self.backend = 'projectq'
        self.seed = seed
        self.file = file
        print("Initialize finished! Time: %.2f s" % self.timer.runtime(), file=self.file)
        sys.stdout.flush()


    def generate_circuit(self, molecule, seed=1202):
        self.n_qubits = molecule.n_qubits
        self.n_electrons = molecule.n_electrons
        self.encoder = build_encoder(num_bit=self.n_qubits, num_electrons=self.n_electrons)
        
        self.ansatz = build_ansatz(num_bit=self.n_qubits, num_electrons=self.n_electrons)
        self.hamiltonian = Hamiltonian(get_qubit_hamiltonian(molecule))
        self.circuit = self.encoder + self.ansatz


    def optimize_using_scipy(self, target):
        self.simulator = Simulator(self.backend, self.n_qubits, seed=1202)
        grad_ops = self.simulator.get_expectation_with_grad(
            self.hamiltonian,
            self.circuit)
        init_gauss = np.zeros(len(self.ansatz.params_name))


        self.res = minimize(func, init_gauss,
                            args=(grad_ops, self.file, target, False),
                            method='BFGS',
                            jac=True,   
                            tol=1e-6
                            )

        
    def get_energy(self):
        param_dict = param2dict(self.circuit.params_name, self.res.x)
        self.simulator.reset()
        self.simulator.apply_circuit(self.circuit, param_dict)
        return self.simulator.get_expectation(self.hamiltonian).real


class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    def run(self, prefix, molecular_file, geom_list):
        prefix = prefix
        molecule = MolecularData(filename=self.work_dir+molecular_file, data_directory='./')
        molecule.load()
        with open(self.work_dir+prefix+'.o', 'a') as f: 
            print('Start case: ', prefix, file=f)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'], file=f)
            vqe = VQEoptimizer(file=f)
            en_list, time_list = [], []
        
            for i in range(len(geom_list)):
                # if prefix == "CH4":
                #     if geom_list[i][1][1][1] == 0.8:
                #         t = vqe.timer.runtime()
                #         en = 0
                #         print('Time: %i hrs %i mints %.2f sec.' % format_time(t), 'Energy: ', en, file=f)
                #         sys.stdout.flush()
                #         en_list.append(en)
                #         time_list.append(t)
                #         continue
                mol = MolecularData(geometry=geom_list[i],
                                        basis=molecule.basis,
                                        charge=molecule.charge,
                                        multiplicity=molecule.multiplicity,
                                        filename=self.work_dir+molecular_file, 
                                        data_directory='./')
                mol = run_pyscf(mol, run_scf=False, run_ccsd=False, run_fci=True, verbose=False)
                print(f"{geom_list[i]}, FCI baseline:{mol.fci_energy}")
                vqe.generate_circuit(mol)
                vqe.optimize_using_scipy(mol.fci_energy)
                t = vqe.timer.runtime()
                en = vqe.get_energy()
                print('Time: %i hrs %i mints %.2f sec.' % format_time(t), 'Energy: ', en, file=f)
                sys.stdout.flush()
                en_list.append(en)
                time_list.append(t)
            
            
            print('Optimization completed. Time: %i hrs %i mints %.2f sec.' % format_time(vqe.timer.runtime()), file=f)

        if len(en_list) == len(geom_list) and len(time_list) == len(geom_list):
            return en_list, time_list#, nparam_list
        else:
            raise ValueError('data lengths are not correct!')

        


class Plot:
    def plot(self, prefix, blen_range, energy, time):
        x = blen_range
        data_time = time
        data_en = energy

        figen, axen = plt.subplots()
        axen.plot(x, data_en)
        axen.set_title('Energy')
        axen.set_xlabel('Bond Length')
        axen.set_ylabel('Energy')
        figen.savefig(f'{prefix}figure_energy.png')

        figtime, axtime = plt.subplots()
        axtime.plot(x, data_time)
        axtime.set_title('Time')
        axtime.set_xlabel('Bond Length')
        axtime.set_ylabel('Time')
        figtime.savefig(f'{prefix}figure_time.png')

        plt.close()
