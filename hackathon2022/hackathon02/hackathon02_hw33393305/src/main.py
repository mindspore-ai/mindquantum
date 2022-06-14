import os
import sys
from hiqfermion.drivers import MolecularData
from openfermionpyscf import run_pyscf
import time
import numpy as np
import matplotlib.pyplot as plt
from mindquantum import Simulator, Hamiltonian, X, Circuit, RY, get_qubit_hamiltonian
from scipy.optimize import minimize


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


def func(x, grad_ops, file, show_iter_val=False):

    f, g = grad_ops(x)
    if show_iter_val:
        print(np.squeeze(f), file=file)
        sys.stdout.flush()
    return np.real(np.squeeze(f)), np.squeeze(g)


def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


class VQEoptimizer:
    def __init__(self, molecule=None, amp_th=0, seed=1202, file=None):
        self.timer = Timer()
        self.molecule = molecule
        self.amp_th = amp_th
        self.backend = 'projectq'
        self.seed = seed
        self.file = file
        self.init_amp = []

        if molecule != None:
            self.generate_circuit(molecule)

        print("Initialize finished! Time: %.2f s" % self.timer.runtime(),
              file=self.file)
        sys.stdout.flush()

    def generate_circuit(self, molecule=None, seed=1202):
        if molecule == None:
            molecule = self.molecule
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

        # qucc一阶
        ansatz_circuit = Circuit()
        for i in range(molecule.n_electrons, molecule.n_qubits):
            for j in range(molecule.n_electrons):
                ansatz_circuit += X.on(j, i)
                ansatz_circuit += RY(f'p{i}_{j}').on(i, j)
                ansatz_circuit += X.on(j, i)
        
        # qucc二阶，消除了空轨道上的对易门
        for ry_ in range(molecule.n_electrons, molecule.n_qubits-1):
            for i in range(ry_ + 1, molecule.n_qubits):
                for j in range(molecule.n_electrons):
                    for k in range(j + 1, molecule.n_electrons):
                        if (j==0 and k==1):
                            ansatz_circuit += X.on(k, j)
                            ansatz_circuit += X.on(k)
                            ansatz_circuit += X.on(i, ry_)
                            ansatz_circuit += X.on(i)
                            ansatz_circuit += X.on(j, ry_)
                            ansatz_circuit += RY(f'p{ry_}_{i}_{j}_{k}').on(
                                ry_, ctrl_qubits=[i, j, k])
                            ansatz_circuit += X.on(j, ry_)
                            ansatz_circuit += X.on(k)
                            ansatz_circuit += X.on(k, j)
                        elif (j==molecule.n_electrons-2 and k==j+1):
                            ansatz_circuit += X.on(k, j)
                            ansatz_circuit += X.on(k)
                            ansatz_circuit += X.on(j, ry_)
                            ansatz_circuit += RY(f'p{ry_}_{i}_{j}_{k}').on(ry_, ctrl_qubits=[i, j, k])
                            ansatz_circuit += X.on(j, ry_)
                            ansatz_circuit += X.on(i)
                            ansatz_circuit += X.on(i, ry_)
                            ansatz_circuit += X.on(k)
                            ansatz_circuit += X.on(k, j)
                        else:
                            ansatz_circuit += X.on(k, j)
                            ansatz_circuit += X.on(k)
                            ansatz_circuit += X.on(j, ry_)
                            ansatz_circuit += RY(f'p{ry_}_{i}_{j}_{k}').on(ry_, ctrl_qubits=[i, j, k])
                            ansatz_circuit += X.on(j, ry_)
                            ansatz_circuit += X.on(k)
                            ansatz_circuit += X.on(k, j)
        

        self.circuit += ansatz_circuit
        self.n_qubits = molecule.n_qubits
        self.params_name = ansatz_circuit.params_name
        self.simulator = Simulator(self.backend, self.n_qubits, seed)
        self.hamiltonian = get_qubit_hamiltonian(molecule)


    def optimize(self,
                 mol,
                 operator=None,
                 circuit=None,
                 init_amp=[],
                 method='bfgs',
                 maxstep=200,
                 iter_info=False):
        if operator == None:
            operator = self.hamiltonian
        if circuit == None:
            circuit = self.circuit
        if np.array(init_amp).size == 0:
            init_amp = self.init_amp

        n_paras = len(self.circuit.params_name)
        init_amp = [1e-6 for i in range(n_paras)]

        grad_ops = self.simulator.get_expectation_with_grad(
            Hamiltonian(operator), circuit)

        # minimize优化器
        self.res = minimize(func, init_amp,
                            args=(grad_ops, self.file, iter_info),
                            method='bfgs',
                            jac=True,
                            tol=1e-3
                            ).x


class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    def run(self, prefix, molecular_file, geom_list):
        prefix = prefix
        molecule = MolecularData(filename=self.work_dir + molecular_file)
        molecule.load()

        with open(self.work_dir + prefix + '.o', 'a') as f:
            print(f)
            print('Start case: ', prefix, file=f)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'], file=f)
            vqe = VQEoptimizer(file=f)
            en_list, time_list = [], []

            for i in range(len(geom_list)):
                mol0 = MolecularData(geometry=geom_list[i],
                                     basis=molecule.basis,
                                     charge=molecule.charge,
                                     multiplicity=molecule.multiplicity)
                mol = run_pyscf(mol0, run_scf=0, run_ccsd=0, run_fci=0)
                vqe.generate_circuit(mol)
                vqe.optimize(mol)
                param_dict = param2dict(vqe.circuit.params_name, vqe.res)

                vqe.simulator.apply_circuit(vqe.circuit, param_dict)
                t = vqe.timer.runtime()
                en = vqe.simulator.get_expectation(Hamiltonian(
                    vqe.hamiltonian)).real
                print('Time: %i hrs %i mints %.2f sec.' % format_time(t),
                      'Energy: ',
                      en,
                      file=f)
                sys.stdout.flush()
                en_list.append(en)
                time_list.append(t)

            print('Optimization completed. Time: %i hrs %i mints %.2f sec.' %
                  format_time(vqe.timer.runtime()),
                  file=f)

        if len(en_list) == len(geom_list) and len(time_list) == len(geom_list):
            return en_list, time_list  # , nparam_list
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
        figen.savefig(prefix + '_figure_energy.png')

        figtime, axtime = plt.subplots()
        axtime.plot(x, data_time)
        axtime.set_title('Time')
        axtime.set_xlabel('Bond Length')
        axtime.set_ylabel('Time')
        figtime.savefig(prefix + '_figure_time.png')

        plt.close()
