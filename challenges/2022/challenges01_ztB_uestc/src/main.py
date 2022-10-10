import os
os.environ['OMP_NUM_THREADS'] = '2'
import sys
import time
import numpy as np
import mindspore as ms
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

from mindquantum import Simulator, Hamiltonian, X, Circuit
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
from mindquantum.core.operators.utils import get_fermion_operator
from transform import jordan_wigner
from mindquantum.third_party.interaction_operator import InteractionOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from uccsd_generator import Uccsd_Generator
from qucc import QubitUCCAnsatz

class Timer:
    def __init__(self, t0=0.0):
        self.start_time = time.time()
        self.t0 = t0

    def runtime(self):
        return time.time() - self.start_time + self.t0

    def resetime(self):
        self.start_time = time.time()

def format_time(t):
    hh = t // 3600
    mm = (t - 3600 * hh) // 60
    ss = t % 60
    return hh, mm, ss


def func(x, grad_ops, show_iter_val=False):
    f, g = grad_ops(x)
    if show_iter_val:
        print(np.real(np.squeeze(f)))
    return np.real(np.squeeze(f)), np.squeeze(g)

def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict

def load_molecular_file(molecular_file):
    molecule = MolecularData(filename=molecular_file, data_directory='./src/hdf5files/')
    molecule = MolecularData(geometry=molecule.geometry, 
                            basis=molecule.basis, 
                            multiplicity=molecule.multiplicity,  
                            charge=molecule.charge, 
                            filename=molecular_file,
                            data_directory='./src/hdf5files/')
    return molecule

class VQEoptimizer:
    def __init__(self, molecule=None, amp_th=0, seed=1, file=None):
        self.molecule = molecule
        self.amp_th = amp_th
        self.backend = 'projectq'
        self.seed = seed
        self.file = file
        self.init_amp = []
        self.en = None

        if molecule != None:
            self.generate_circuit(molecule)

    def generate_circuit(self, molecule=None, seed=765):
        if molecule == None:
            molecule = self.molecule
        self.en = molecule.fci_energy
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])
        
        # H6采用qUCC进行求解，LiH采用UCCSD
        if molecule.name == 'H6_sto-3g_singlet': #  and molecule.geometry[1][1][2] < 0.91:
            qucc = QubitUCCAnsatz(molecule, molecule.n_qubits,
                          molecule.n_electrons, trotter_step=1, th=0.0046)
            ansatz_circuit = qucc.circuit
            self.init_amp = [0 for _ in range(qucc.num_params)]
            # self.init_amp = qucc.params
            self.n_qubits = molecule.n_qubits
            self.n_electrons = molecule.n_electrons
            
            ham_of = molecule.get_molecular_hamiltonian()
            inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
            ham_hiq = get_fermion_operator(inter_ops)
            
            # 剔除哈密顿量中系数小的项
            d = {}
            for k,v in ham_hiq.terms.items():
                if abs(v) >= 0.005:
                    d[k]=v
            ham_hiq.terms = d
        
            qubit_hamiltonian = jordan_wigner(ham_hiq)
            qubit_hamiltonian.compress()
            self.hamiltonian = qubit_hamiltonian.real
            
        else:
            ansatz_circuit, \
            self.init_amp, \
            self.params_name, \
            self.hamiltonian, \
            self.n_qubits, \
            self.n_electrons = Uccsd_Generator(molecule, self.amp_th, 'JW').generate_uccsd

        self.circuit += ansatz_circuit 
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

    def optimize(self, operator=None, circuit=None, init_amp=[],
                 method='bfgs', maxstep=300, iter_info=False, gtol=1e-2):
        if operator == None:
            operator = self.hamiltonian
        if circuit == None:
            circuit = self.circuit
        if np.array(init_amp).size == 0:
            init_amp = self.init_amp

        grad_ops = self.simulator.get_expectation_with_grad(Hamiltonian(self.hamiltonian), circuit)
        net = MQAnsatzOnlyLayer(grad_ops)
        net.weight = ms.Parameter(ms.Tensor(init_amp, net.weight.dtype))
        optimizer = ms.nn.Adagrad(net.trainable_params(), learning_rate=0.22)
        train_net = ms.nn.TrainOneStepCell(net, optimizer)
        
        eps = 0.0016
        energy_err = eps * 1000
        iter_idx = 0
        energy = 0
        while abs(energy_err) >= eps and iter_idx <= maxstep:
            energy = train_net().asnumpy()

            energy_err = abs(energy - self.en)
            iter_idx += 1
            
        self.res = energy[0]

class Main:
    def __init__(self):
        super().__init__()

        self.work_dir = './src/'
        self.vqe = None


    def run(self, prefix, molecular_file):
        ath = 0.0066

        molecule = load_molecular_file(molecular_file)

        self.vqe = VQEoptimizer(amp_th=ath)

        mol = run_pyscf(molecule, run_scf=0, run_ccsd=1, run_fci=1)
        self.vqe.generate_circuit(mol)

        self.vqe.optimize(gtol=0.0178)
        en = self.vqe.res

        return en
