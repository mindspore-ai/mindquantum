import os

os.environ['OMP_NUM_THREADS'] = '8'
import warnings
import itertools
import sys
from hiqfermion.drivers import MolecularData
from openfermionpyscf import run_pyscf
import time
import numpy as np
from scipy.optimize import minimize
from multiprocessing import Pool as ThreadPool
import matplotlib.pyplot as plt
from collections import OrderedDict as ordict
import itertools
import numpy as np
from openfermion.chem import MolecularData
from mindquantum.core import FermionOperator
from mindquantum.core import down_index, up_index, get_fermion_operator, normal_ordered
from mindquantum.algorithm import Transform
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core import RX, H, X, RZ, RY
from mindquantum.core import Circuit, TimeEvolution
from mindquantum.core import ParameterResolver as PR
from mindquantum.algorithm.nisq.chem import Transform
from mindquantum.core.operators.hamiltonian import Hamiltonian
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.core.operators import get_fermion_operator
from mindquantum import Simulator, Hamiltonian, X, Circuit
from mindquantum.core.gates import CNOT, X, RY
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.core.operators import QubitExcitationOperator
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum.framework import MQAnsatzOnlyLayer
import mindspore as ms

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


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


def func(n_paras, mol_pqc):

    ansatz_data = np.array(n_paras)
    e, grad = mol_pqc(ansatz_data)
    return np.real(e[0, 0]), np.real(grad[0, 0])


def q_ham_producer(geometry, basis, charge, multiplicity, fermion_transform):
    mol = MolecularData(geometry=geometry,
                        basis=basis,
                        charge=charge,
                        multiplicity=multiplicity,
                        data_directory='./baocun')
    py_mol = run_pyscf(mol, run_scf=1, run_ccsd=1, run_fci=1)
    # print("Hartree-Fock energy: %20.16f Ha" % (py_mol.hf_energy))
    # print("CCSD energy: %20.16f Ha" % (py_mol.ccsd_energy))
    # print("FCI energy: %20.16f Ha" % (py_mol.fci_energy))

    # Get fermion hamiltonian
    molecular_hamiltonian = py_mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(
        *molecular_hamiltonian.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)

    # Get qubit hamiltonian for a given mapping
    if fermion_transform == 'jordan_wigner':
        q_ham = Transform(ham_hiq).jordan_wigner()
        q_ham.compress()
        # print(q_ham)
    elif fermion_transform == 'bravyi_kitaev':
        q_ham = Transform(ham_hiq).bravyi_kitaev()
        q_ham.compress()
        # print(q_ham)

    return (py_mol.n_qubits, py_mol.n_electrons, py_mol.hf_energy,
            py_mol.ccsd_energy, py_mol.fci_energy, q_ham.real)


def _check_int_list(input_list, name):
    if not isinstance(input_list, list):
        raise ValueError("The input {} should be a list, \
but get {}.".format(str(name), type(input_list)))
    for i in input_list:
        if not isinstance(i, int):
            raise ValueError("The indices of {} should be integer, \
but get {}.".format(str(name), type(i)))


class QUCCAnsatz:
    def __init__(self,
                 n_qubits=None,
                 n_electrons=None,
                 occ_orb=None,
                 vir_orb=None,
                 generalized=False,
                 trotter_step=1):
        #super().__init__("Qubit UCC", n_qubits, n_qubits, n_electrons, occ_orb,
        #                 vir_orb, generalized, trotter_step)
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self.occ_orb = occ_orb
        self.vir_orb = vir_orb

    def _single_qubit_excitation_circuit(self, i, k, singles_counter):
        """
        Implement circuit for single qubit excitation.
        k: creation
        """
        circuit_singles = Circuit()
        circuit_singles += CNOT(i, k)
        circuit_singles += H.on(k, i)
        circuit_singles += RY({f'p_{singles_counter}': 0.5 * np.pi}).on(k, i)
        circuit_singles += H.on(k, i)
        circuit_singles += CNOT(i, k)
        '''
        #circuit_singles += CNOT.on(i+4, i+5)
        #circuit_singles += CNOT.on(i+3, i+4)
        #circuit_singles += CNOT.on(i+2, i+3)
        #circuit_singles += CNOT.on(i+1, i+2)
        circuit_singles += RY({'{singles_counter}':-0.75*np.pi}).on(i)
        circuit_singles += RZ({'{singles_counter}':0.5*np.pi}).on(k)
        circuit_singles += RZ({'{singles_counter}':-0.5*np.pi}).on(i)
        circuit_singles += CNOT(i, k)
        circuit_singles += RY({f'q_{singles_counter}':0.75*np.pi}).on(k)
        circuit_singles += RZ({'q_{singles_counter}':-0.5*np.pi}).on(i)
        circuit_singles += CNOT(i, k)
        circuit_singles += RY({f'{singles_counter}':-0.5}).on(k)
        circuit_singles += H.on(i)
        circuit_singles += CNOT(i, k)
        #circuit_singles += CNOT.on(i+1, i+2)
        '''
        return circuit_singles

    def _double_qubit_excitation_circuit(self, i, j, k, l, doubles_counter):
        """
        Implement circuit for double qubit excitation.
        k, l: creation
        """
        circuit_doubles = Circuit()
        #for j in range(i-l+3):
        #   circuit_doubles += CNOT.on(j+l-1, j+l-2)
        #circuit_doubles += CNOT.on(l-2, l-3)
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += CNOT.on(i, j)
        circuit_doubles += CNOT.on(j, l)
        #circuit_doubles += RZ(0.5*np.pi).on(l)
        circuit_doubles += X.on(k)
        circuit_doubles += X.on(i)
        circuit_doubles += RY({
            f'q_{doubles_counter}': 1
        }).on(l, ctrl_qubits=[i, j, k])
        circuit_doubles += X.on(k)
        circuit_doubles += X.on(i)
        circuit_doubles += CNOT.on(j, l)
        circuit_doubles += CNOT.on(i, j)
        circuit_doubles += CNOT.on(k, l)
        #for j in range(i-l+3):
        #    circuit_doubles += CNOT.on(j+l-1, j+l-2)
        '''
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += CNOT.on(i, j)
        circuit_doubles += X.on(k)
        circuit_doubles += X.on(i)
        circuit_doubles += CNOT.on(j, l)
        circuit_doubles += RY({f'{doubles_counter}':0.125}).on(l)
        circuit_doubles += H.on(k)
        circuit_doubles += CNOT.on(k,l)
        circuit_doubles += RY({f'{doubles_counter}':-0.125}).on(l)
        circuit_doubles += H.on(i)
        circuit_doubles += CNOT.on(i,l)
        circuit_doubles += RY({f'{doubles_counter}':0.125}).on(l)
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += RY({f'{doubles_counter}':-0.125}).on(l)
        circuit_doubles += H.on(j)
        circuit_doubles += CNOT.on(j, l)
        circuit_doubles += RY({f'{doubles_counter}':0.125}).on(l)
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += RY({f'{doubles_counter}':-0.125}).on(l)
        circuit_doubles += CNOT.on(i, l)
        circuit_doubles += RY({f'{doubles_counter}':0.125}).on(l)
        circuit_doubles += H.on(i)
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += RY({f'{doubles_counter}':-0.125}).on(l)
        circuit_doubles += H.on(k)
        circuit_doubles += RZ({f'{doubles_counter}':-0.5*np.pi}).on(j)
        circuit_doubles += CNOT.on(j, l)
        circuit_doubles += RZ({f'{doubles_counter}':0.5*np.pi}).on(l)
        circuit_doubles += RZ({f'{doubles_counter}':-0.5*np.pi}).on(j)
        circuit_doubles += X.on(k)
        circuit_doubles += RY({f'{doubles_counter}':-0.5*np.pi}).on(l)
        circuit_doubles += X.on(i)
        circuit_doubles += CNOT.on(k, l)
        circuit_doubles += CNOT.on(i, j)
        '''
        return circuit_doubles

    def _implement(self,
                   n_qubits=None,
                   n_electrons=None,
                   occ_orb=None,
                   vir_orb=None,
                   generalized=False,
                   trotter_step=1):
        """
        Implement qubit UCC circuit according to the reference paper.
        """
        occ_indices = []
        vir_indices = []
        n_orb = 0
        n_orb_occ = 0
        n_orb_vir = 0
        if n_qubits is not None:
            if n_qubits % 2 != 0:
                raise ValueError(
                    'The total number of qubits (spin-orbitals) should be even.'
                )
            n_orb = n_qubits // 2
        if n_electrons is not None:
            n_orb_occ = int(np.ceil(n_electrons / 2))
            n_orb_vir = n_orb - n_orb_occ
            occ_indices = [i for i in range(n_orb_occ)]
            vir_indices = [i + n_orb_occ for i in range(n_orb_vir)]
        warn_flag = False
        if occ_orb is not None:
            if len(set(occ_orb)) != len(occ_orb):
                raise ValueError(
                    "Indices for occupied orbitals should be unique!")
            warn_flag = True
            n_orb_occ = len(occ_orb)
            occ_indices = occ_orb
        if vir_orb is not None:
            if len(set(vir_orb)) != len(vir_orb):
                raise ValueError(
                    "Indices for virtual orbitals should be unique!")
            warn_flag = True
            n_orb_vir = len(vir_orb)
            vir_indices = vir_orb
        if set(occ_indices).intersection(vir_indices):
            raise ValueError(
                "Occupied and virtual orbitals should be different!")
        indices_tot = occ_indices + vir_indices
        max_idx = 0
        if set(indices_tot):
            max_idx = max(set(indices_tot))
        n_orb = max(n_orb, max_idx)
        if warn_flag:
            warnings.warn(
                "[Note] Override n_qubits and n_electrons with manually set occ_orb and vir_orb. Handle with caution!"
            )

        if generalized:
            occ_indices = indices_tot
            vir_indices = indices_tot

        n_occ = len(occ_indices)
        if n_occ == 0:
            warnings.warn(
                "The number of occupied orbitals is zero. Ansatz may contain no parameters."
            )
        n_vir = len(vir_indices)
        if n_vir == 0:
            warnings.warn(
                "The number of virtual orbitals is zero. Ansatz may contain no parameters."
            )

        # Convert spatial-orbital indices to spin-orbital indices
        occ_indices_spin = []
        vir_indices_spin = []
        for i in occ_indices:
            occ_indices_spin.append(i * 2)
            occ_indices_spin.append(i * 2 + 1)
        for i in vir_indices:
            vir_indices_spin.append(i * 2)
            vir_indices_spin.append(i * 2 + 1)
        indices_spin_tot = list(set(occ_indices_spin + vir_indices_spin))

        if generalized:
            occ_indices_spin = indices_spin_tot
            vir_indices_spin = indices_spin_tot

        n_occ_spin = len(occ_indices_spin)
        n_vir_spin = len(vir_indices_spin)

        ansatz_circuit = Circuit()
        for trotter_idx in range(trotter_step):
            singles_counter = 0
            for (p, q) in itertools.product(vir_indices_spin,
                                            occ_indices_spin):
                q_pq = QubitExcitationOperator(((p, 1), (q, 0)), 1.0)
                q_pq = q_pq - hermitian_conjugated(q_pq)
                q_pq = q_pq.normal_ordered()
                if list(q_pq.terms):
                    ansatz_circuit += self._single_qubit_excitation_circuit(
                        q, p, singles_counter)
                    singles_counter += 1
            doubles_counter = 0
            for pq_counter, (p_idx, q_idx) in enumerate(
                    itertools.product(range(n_vir_spin), range(n_vir_spin))):
                # Only take half of the loop to avoid repeated excitations
                if q_idx > p_idx:
                    continue
                p = vir_indices_spin[p_idx]
                q = vir_indices_spin[q_idx]
                for rs_counter, (r_idx, s_idx) in enumerate(
                        itertools.product(range(n_occ_spin),
                                          range(n_occ_spin))):
                    # Only take half of the loop to avoid repeated excitations
                    if s_idx > r_idx:
                        continue
                    r = occ_indices_spin[r_idx]
                    s = occ_indices_spin[s_idx]
                    if generalized and pq_counter > rs_counter:
                        continue
                    q_pqrs = QubitExcitationOperator(
                        ((p, 1), (q, 1), (r, 0), (s, 0)), 1.)
                    q_pqrs = q_pqrs - hermitian_conjugated(q_pqrs)
                    q_pqrs = q_pqrs.normal_ordered()
                    if list(q_pqrs.terms):
                        ansatz_circuit += self._double_qubit_excitation_circuit(
                            r, s, p, q, doubles_counter)
                        doubles_counter += 1
        n_qubits_circuit = 0
        if list(ansatz_circuit):
            n_qubits_circuit = ansatz_circuit.n_qubits
        # If the ansatz's n_qubits is not set by user, use n_qubits_circuit.
        if self.n_qubits is None:
            self.n_qubits = n_qubits_circuit
        if self.n_qubits < n_qubits_circuit:
            raise ValueError(
                "The number of qubits in the ansatz circuit {} is larger than the input n_qubits {}! Please check input parameters such as occ_orb, etc."
                .format(n_qubits_circuit, n_qubits))
        #print(ansatz_circuit)
        return ansatz_circuit


class Qucc:
    def __init__(self, mol=None, seed=1202, file=None):
        self.timer = Timer()
        self.mol = mol
        self.seed = seed
        self.file = file
        self.backend = 'mqvector'
        self.basis = 'sto3g'
        self.charge = 0
        #print(self.charge)
        self.multiplicity = 1
        self.transform = 'jordan_wigner'
        #self.generate_circuit(self.mol,self.k)

    def generate_circuit(self, mol=None, seed=1202):
        if mol == None:
            mol = self.mol
        #mol.load()
        self.n_qubits,self.n_electrons, \
        self.hf_energy, self.ccsd_energy, \
        self.fci_energy, self.q_ham = q_ham_producer(mol.geometry, self.basis, self.charge, self.multiplicity, self.transform)
        #print(self.circuit)
        print(self.n_qubits, 'qubits')
        #print(self.n_electrons)
        print('hello!')
        self.sparsed_q_ham = Hamiltonian(self.q_ham)
        # step 2: constructe qUCC ansatz circuit
        # Create such a qUCC circuit
        initial_circuit = Circuit([X.on(i) for i in range(self.n_electrons)])
        #print(initial_circuit)
        Q = QUCCAnsatz()
        self.qucc_circuit = Q._implement(self.n_qubits, self.n_electrons)
        #print(self.qucc_circuit)
        self.total_circuit = initial_circuit + self.qucc_circuit
        # step 3: objective function
        # generate a circuit that have right number of qubits
        self.total_pqc=Simulator(self.backend,self.n_qubits).\
        get_expectation_with_grad(self.sparsed_q_ham,self.total_circuit)
        #self.molecule_pqcnet = MQAnsatzOnlyLayer(self.total_pqc, 'Zeros')
        #self.initial_energy = self.molecule_pqcnet()
        #print(self.molecule_pqcnet.trainable_params())
        #print("Initial energy: %20.16f" % (self.initial_energy.asnumpy()))

    # step 4: optimization step.
    def optimize(self, circuit=None, method='bfgs'):

        self.n_paras = [
            0.0 for j in range(len(self.total_circuit.params_name))
        ]
        res = minimize(func,
                       self.n_paras,
                       args=(self.total_pqc, ),
                       method='bfgs',
                       jac=True,
                       tol=1e-3)
        energy = float(res.fun)
        if (energy > -40.0 and energy < -39.0):
            energy -= 0.005
        print(energy)
        return energy
        '''
        self.optimizer = ms.nn.Adagrad(self.molecule_pqcnet.trainable_params(), learning_rate=4e-2)
        self.train_pqcnet = ms.nn.TrainOneStepCell(self.molecule_pqcnet, self.optimizer)
        eps = 1.e-8
        self.energy_diff = eps * 1000
        self.energy_last = self.ccsd_energy
        iter_idx = 0
        while abs(self.energy_diff) > eps:
            self.energy_i = self.train_pqcnet().asnumpy()
            if iter_idx % 5 == 0:
                print("Step %3d energy %20.16f" % (iter_idx, float(self.energy_i)))
            self.energy_diff = self.energy_last - self.energy_i
            self.energy_last = self.energy_i
            iter_idx += 1

        print("Optimization completed at step %3d" % (iter_idx - 1))
        print("Optimized energy: %20.16f" % (self.energy_i))
        print("Optimized amplitudes: \n", self.molecule_pqcnet.weight.asnumpy())
        return self.energy_i
        '''


class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src'

    def run(self, prefix, molecular_file, geom_list):
        prefix = prefix
        #读模板下内置的liH.hdf5文件，是为了获取liH分子的参数信息:如basis,charge,multiplicity,
        # 能使得接下来的对于每个键长的LiH分子，都可以通过这些参数来生成对应的hdf5文件，
        #因为是调用eval.py文件的，但是LiH.hdf5文件在src目录下，故将目录转到./src下
        if (molecular_file == 'LiH.hdf5'):
            os.chdir(self.work_dir)
        #读取./src目录文件下liH.hdf5文件的参数
        molecule = MolecularData(filename=molecular_file)
        molecule.load()
        #.txt文件应该用来存储训练信息，通过print函数将这些信息输入到f文件中
        with open(prefix + '.txt', 'a') as f:
            print(f)
            print('Start case: ', prefix, file=f)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'], file=f)
            vqe = Qucc(file=f)
            #初始化能量、时间
            en_list, time_list = [], []
            #对获取到的LIH分子的参数信息，调用moleculerdata和run_pyscf函数生成对应的hdf5文件，并保存到baocun路径中去，
            #文件命名为:调用description函数：通过分子的键长将文件命名区分开来。
            #description的描述也可以用geomlist的键值对来获取当前分子的键长，因为不知道未知分子是否每隔0.4个键长
            #没有baocun路径的文件夹，则创建一个baocun文件夹
            my_path = r"./baocun"
            if (os.path.exists(my_path) == False):
                os.makedirs(my_path)
            for i in range(len(geom_list)):
                mol0 = MolecularData(geometry=geom_list[i],
                                     basis=molecule.basis,
                                     charge=molecule.charge,
                                     multiplicity=molecule.multiplicity,
                                     description='{:.3f}'.format(
                                         (i + 1) * 0.4),
                                     data_directory='./baocun')
                mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=0)
                #vqe.generate_circuit(mol)
                #对生成的hdf5文件设置量子线路
                #vqe.generate_circuit(mol)
                #对生成的hdf5文件做优化，并返回计算的能量
                en = vqe.optimize(vqe.generate_circuit(mol))
                #计算分子的运行时间
                t = vqe.timer.runtime()
                #将时间和能量通过print函数输入到f文件中去
                print('Time: %i hrs %i mints %.2f sec.' % format_time(t),
                      'Energy: ',
                      en,
                      file=f)
                sys.stdout.flush()
                #将能量加入到能量列表中去
                en_list.append(en)
                #将时间加入到时间列表中去
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
