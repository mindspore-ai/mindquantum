import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from hiqfermion.drivers import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum import Simulator, Hamiltonian, X, Circuit
from mindquantum.core.circuit import Circuit
import order_effectiave_uccsd as OEU

import os
import sys
import time


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
        print(np.squeeze(f).real)
        sys.stdout.flush()
    return np.real(np.squeeze(f)), np.squeeze(g)


def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


"""
1. 从General_ansatz中产生ansatz依然是基于UCCSD，但是我们对其进行了改进，重新命名为order-effectiave-UCCSD。
2. 这个ansaztz对所有分子是通用的，具有可拓展行，。
3. order-effectiave-UCCSD 极大的减少了量子线路中所要优化的参数和量子门的数量
    对于LiH分子: 从44个优化参数减到只需要优化7个参数
    对于CH4分子: 从230个优化参数减到只需要优化68个参数
"""


class VQEoptimizer:
    def __init__(self, molecule=None, amp_th=-1, seed=1202, file=None):
        self.timer = Timer()
        self.molecule = molecule
        self.amp_th = amp_th
        self.backend = 'mqvector'
        self.seed = seed
        self.file = file
        self.init_amp = []

        print("Initialize finished! Time: %.2f s" % self.timer.runtime())
        sys.stdout.flush()

    def generate_circuit(self, prefix, blen, molecule=None, seed=1202):
        if molecule == None:
            molecule = self.molecule

        self.blen = blen
        self.prefix = prefix
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

        # 因为LiH特殊的分子轨道结构，非占据态的二，三轨道电子云有很小的重叠，所以就不考虑这两个轨道

        ansatz_circuit, \
        self.init_amp, \
        self.params_name, \
        self.hamiltonian, \
        self.n_qubits, \
        self.n_electrons = OEU.generate_uccsd(molecule, self.amp_th, prefix, blen)

        self.circuit += ansatz_circuit
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

    def optimize(self,
                 operator=None,
                 circuit=None,
                 init_amp=[],
                 method='SLSQP',
                 maxstep=200,
                 iter_info=False):
        if operator == None:
            operator = self.hamiltonian
        if circuit == None:
            circuit = self.circuit
        if np.array(init_amp).size == 0:
            init_amp = self.init_amp

        grad_ops = self.simulator.get_expectation_with_grad(
            Hamiltonian(operator), circuit)

        # 因为比赛比的是达到化学精度的前提下，用尽可能短的时间，所以在优化LiH和CH4的时候，我们调节优化器的终止的精度范围。
        if self.prefix == 'LiH':
            self.res = minimize(func,
                                init_amp,
                                args=(grad_ops, self.file, iter_info),
                                method=method,
                                jac=True,
                                tol=0.0001)
        else:
            self.res = minimize(func,
                                init_amp,
                                args=(grad_ops, self.file, iter_info),
                                method=method,
                                jac=True,
                                tol=0.00016)


class Main:
    def __init__(self):
        super().__init__()
        #self.work_dir = './src/'
        self.work_dir = os.path.dirname(os.path.abspath(__file__)) + '/'

    def run(self, prefix, molecular_file, geom_list):
        prefix = prefix
        molecule = MolecularData(filename=self.work_dir + molecular_file)
        molecule.load()

        with open(self.work_dir + prefix + '.o', 'a') as f:

            print('Start case: ', prefix)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'])
            vqe = VQEoptimizer(file=f)
            en_list, time_list = [], []

            for i in range(len(geom_list)):
                mol0 = MolecularData(geometry=geom_list[i],
                                     basis=molecule.basis,
                                     charge=molecule.charge,
                                     multiplicity=molecule.multiplicity)
                mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=1)
                blen = i
                vqe.generate_circuit(prefix, blen, mol)
                vqe.optimize()
                param_dict = param2dict(vqe.circuit.params_name, vqe.res.x)

                vqe.simulator.apply_circuit(vqe.circuit, param_dict)
                t = vqe.timer.runtime()
                en = vqe.simulator.get_expectation(Hamiltonian(
                    vqe.hamiltonian)).real
                print('Time: %i hrs %i mints %.2f sec.' % format_time(t),
                      'Energy: ', en)
                print("The error is: ", (mol0.fci_energy - en))
                sys.stdout.flush()
                en_list.append(en)
                time_list.append(t)

            print('Optimization completed. Time: %i hrs %i mints %.2f sec.' %
                  format_time(vqe.timer.runtime()))

        if len(en_list) == len(geom_list) and len(time_list) == len(geom_list):
            return en_list, time_list  #, nparam_list
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
        figen.savefig('figure_energy.png')

        figtime, axtime = plt.subplots()
        axtime.plot(x, data_time)
        axtime.set_title('Time')
        axtime.set_xlabel('Bond Length')
        axtime.set_ylabel('Time')
        figtime.savefig('figure_time.png')

        plt.close()
