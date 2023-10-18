import os

os.environ['OMP_NUM_THREADS'] = '6'
import sys
#from hiqfermion.drivers import MolecularData
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import time
import numpy as np
from mindquantum import Simulator, Hamiltonian, X, Circuit
from scipy.optimize import minimize
from mindquantum.algorithm import generate_uccsd

from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
from mindquantum.algorithm.nisq.chem import uccsd_singlet_generator, uccsd_singlet_get_packed_amplitudes
from mindquantum.core.operators import TimeEvolution

import mindspore as ms
from mindquantum.framework import MQAnsatzOnlyLayer
from mindspore import Parameter

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

import matplotlib.pyplot as plt


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


def optimize_mindspore(molecule_pqc, ini_amp, total_iter=100):

    if len(ini_amp) == 0:
        molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc, 'Zeros')
    else:
        molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc)
        molecule_pqcnet.weight = Parameter(
            ms.Tensor(ini_amp, molecule_pqcnet.weight.dtype))

    initial_energy = molecule_pqcnet()
    optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(),
                              learning_rate=0.075)
    train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

    eps = 1e-5
    energy_diff = eps * 1000
    energy_last = initial_energy.asnumpy() + energy_diff
    iter_idx = 0
    while abs(energy_diff) > eps and iter_idx < total_iter:
        energy_i = train_pqcnet().asnumpy()
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1

    optimized_params = molecule_pqcnet.weight.asnumpy()
    return optimized_params


def get_ccsd_ini_amp(molecule):
    hamiltonian = get_qubit_hamiltonian(molecule)
    ucc_fermion_ops = uccsd_singlet_generator(molecule.n_qubits,
                                              molecule.n_electrons,
                                              anti_hermitian=True)
    ucc_qubit_ops = Transform(ucc_fermion_ops).jordan_wigner()
    ansatz_circuit = TimeEvolution(ucc_qubit_ops.imag, 1.0).circuit
    ansatz_parameter_names = ansatz_circuit.params_name

    init_amplitudes_ccsd = uccsd_singlet_get_packed_amplitudes(
        molecule.ccsd_single_amps, molecule.ccsd_double_amps,
        molecule.n_qubits, molecule.n_electrons)
    init_amplitudes_ccsd = [
        init_amplitudes_ccsd[param_i] for param_i in ansatz_parameter_names
    ]

    return init_amplitudes_ccsd, ansatz_circuit, hamiltonian


class VQEoptimizer:
    def __init__(self,
                 molecule=None,
                 amp_th=0,
                 use_ccsd_ini=True,
                 seed=1202,
                 file=None):
        self.timer = Timer()
        self.molecule = molecule
        self.amp_th = amp_th
        self.backend = 'mqvector'
        self.seed = seed
        self.file = file
        self.init_amp = []
        self.use_ccsd_ini = use_ccsd_ini

        if molecule != None:
            self.generate_circuit(molecule)

        print("Initialize finished! Time: %.2f s" % self.timer.runtime(),
              file=self.file)
        sys.stdout.flush()

    def generate_circuit(self, molecule=None, seed=1202):
        if molecule == None:
            molecule = self.molecule
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

        if not self.use_ccsd_ini:
            ansatz_circuit, \
            self.init_amp, \
            self.params_name, \
            self.hamiltonian, \
            self.n_qubits, \
            self.n_electrons = generate_uccsd(molecule, self.amp_th)
        else:
            init_amplitudes_ccsd, ansatz_circuit, hamiltonian = get_ccsd_ini_amp(
                molecule)
            self.init_amp = init_amplitudes_ccsd
            self.hamiltonian = hamiltonian
            self.n_qubits = molecule.n_qubits
            self.n_electrons = molecule.n_electrons

        self.circuit += ansatz_circuit
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

    def optimize(self,
                 operator=None,
                 circuit=None,
                 init_amp=[],
                 method='mindspore_layer',
                 maxstep=200,
                 iter_info=False):
        if operator == None:
            operator = self.hamiltonian.real
        if circuit == None:
            circuit = self.circuit

        if np.array(init_amp).size == 0:
            init_amp = self.init_amp

        molecule_pqc = self.simulator.get_expectation_with_grad(
            Hamiltonian(operator), circuit)

        if method == 'bfgs':
            res = minimize(func,
                           init_amp,
                           args=(molecule_pqc, self.file, iter_info),
                           method=method,
                           jac=True)
            return res.x

        if method == 'mindspore_layer':
            res = optimize_mindspore(molecule_pqc, self.init_amp, maxstep)
            return res


class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    def run(self, prefix, molecular_file, geom_list):
        molecule = MolecularData(filename=self.work_dir + molecular_file)
        molecule.load()
        maxstep = 40
        if prefix == 'LiH':
            maxstep = 100

        if prefix == 'CH4':
            maxstep = 1

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
                mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=0)
                vqe.generate_circuit(mol)
                res = vqe.optimize(method='mindspore_layer', maxstep=maxstep)

                param_dict = param2dict(vqe.circuit.params_name, res)

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
