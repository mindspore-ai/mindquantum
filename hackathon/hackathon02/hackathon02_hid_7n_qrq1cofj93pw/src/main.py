import os, sys
import time
os.environ['OMP_NUM_THREADS'] = '8'

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf

from mindquantum import Circuit, X, Hamiltonian, Simulator
# from mindquantum.algorithm import generate_uccsd
from my_algorithm import generate_uccsd
import mindspore as ms
import mindspore.context as context
from mindspore import Parameter
from mindquantum.framework import MQAnsatzOnlyLayer
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

import numpy as np
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

class VQEoptimizer:
    def __init__(self, molecule=None, amp_th=1e-4, seed=1202, file=None):
        self.timer = Timer()
        self.molecule = molecule
        self.amp_th = amp_th
        self.backend = 'projectq'
        self.seed = seed
        self.file = file
        self.init_amp = []
        self.weight = []

        if molecule != None:
            self.generate_circuit(molecule)

        print("Initialize finished! Time: %.2f s" % self.timer.runtime(), file=self.file)
        sys.stdout.flush()

    def generate_circuit(self, molecule=None, seed=1202):
        if molecule == None:
            molecule = self.molecule
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])
        # 说明，此处对 generate_ccsd 做出修改，参见 my_algorithm 的第149-192行.

        ansatz_circuit, \
        self.init_amp, \
        self.params_name, \
        self.hamiltonian, \
        self.n_qubits, \
        self.n_electrons = generate_uccsd(molecule, th=self.amp_th)
        self.circuit += ansatz_circuit 
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

    def optimize(self, operator=None, circuit=None, init_amp=[], maxstep=500, iter_info=False):
        if operator == None:
            operator = self.hamiltonian
        if circuit == None:
            circuit = self.circuit
        if np.array(init_amp).size == 0:
            init_amp = self.init_amp

        print(circuit.summary(), file=self.file)

        grad_ops = self.simulator.get_expectation_with_grad(Hamiltonian(operator), circuit, parallel_worker=8)
        net = MQAnsatzOnlyLayer(grad_ops, weight='Zeros')
        
        learning_rate = 2.0 / self.n_electrons * 2e-2
        # print("learning rate = ", learning_rate)
        optimizer = ms.nn.Adam(net.trainable_params(), learning_rate=learning_rate)
        train_net = ms.nn.TrainOneStepCell(net, optimizer)
        
        energy_fci = self.molecule.fci_energy
        energy_ccsd = self.molecule.ccsd_energy
        print("ccsd energy: ", energy_ccsd)
        # print("fci energy:", energy_fci, file=self.file)
        eps = 0.0016 # 精度
        energy_diff = eps * 10.0
        energy_best = 0.0  # 最低能量
        amps_best = None   # 最佳幅值

        iter_idx = 0 # 迭代次数
        start_time = time.time() # 起始时间
        while abs(energy_diff) > eps and iter_idx < maxstep:
            energy_i = train_net().asnumpy()[0]
            energy_diff = energy_fci - energy_i

            if energy_i < energy_best: # 记录最小值
                energy_best = energy_i
                amps_best = net.weight.asnumpy() # 记录最佳幅度
            iter_idx += 1
            
            if iter_info:
                print("iter %d spend %6.3f minutes. energy: %10.4f" % (iter_idx, (time.time() - start_time)/60,  energy_i))
            start_time = time.time()
        net.weight = Parameter(ms.Tensor(amps_best, net.weight.dtype))
        self.weight = amps_best


class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    def run(self, prefix, molecular_file, geom_list):
        prefix = prefix
        molecule = MolecularData(filename=self.work_dir+molecular_file)
        molecule.load()

        with open(self.work_dir+prefix+'.o', 'a') as f:
            print(f)
            print('Start case: ', prefix, file=f)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'], file=f)

            vqe = VQEoptimizer(amp_th=0.003733, file=f)
            en_list, time_list = [], []

            for i in range(len(geom_list)):
                mol0 = MolecularData(geometry=geom_list[i],
                                    basis=molecule.basis,
                                    charge=molecule.charge,
                                    multiplicity=molecule.multiplicity)
                mol0.filename = self.work_dir + prefix + "_molecule.tmp"
                mol = run_pyscf(mol0, run_scf=0, run_ccsd=1, run_fci=1)
                vqe.molecule = mol
                vqe.generate_circuit(mol)
                vqe.optimize(iter_info=False)
                param_dict = param2dict(vqe.circuit.params_name, vqe.weight)

                vqe.simulator.apply_circuit(vqe.circuit, param_dict)
                t = vqe.timer.runtime()
                en = vqe.simulator.get_expectation(Hamiltonian(vqe.hamiltonian)).real
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
        axen.plot(x, data_en, 'ro-')
        axen.set_title('Energy')
        axen.set_xlabel('Bond Length')
        axen.set_ylabel('Energy')
        figen.savefig('figure_energy.png')

        figtime, axtime = plt.subplots()
        axtime.plot(x, data_time, 'ro-')
        axtime.set_title('Time')
        axtime.set_xlabel('Bond Length')
        axtime.set_ylabel('Time')
        figtime.savefig('figure_time.png')

        plt.close()


#*** 测试 ***#
def geometry_lih(blens):
    geom = []
    for blen in blens:
        geom.append([('Li', [0, 0, 0]), ('H', [0, 0, blen])])

    return geom

def geometry_ch4(blens):
    geom = []
    for blen in blens:
        geom.append([('C', [0.0, 0.0, 0.0]),
                         ('H', [blen, blen, blen]),
                         ('H', [blen, -blen, -blen]),
                         ('H', [-blen, blen, -blen]),
                         ('H', [-blen, -blen, blen])
                         ])

    return geom

if __name__ == '__main__':
    main = Main()
    plt_handle = Plot()

    molecule1 = 'LiH.hdf5'
    molecule2 = 'CH4.hdf5'

    blen1 = [0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
    # blen2 = [0.4, 0.6, 0.8]
    # blen1 = [4.0]
    blen2 = [0.8, 0.4]

    geom1 = geometry_lih(blen1)
    geom2 = geometry_ch4(blen2)

    en1, time1 = main.run('LiH', molecule1, geom1)
    # en2, time2 = main.run('CH4', molecule2, geom2)

    # plt_handle.plot('LiH', blen1, en1, time1)
