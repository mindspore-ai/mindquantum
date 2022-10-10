import os
os.environ['OMP_NUM_THREADS'] = '4'
import sys
#from hiqfermion.drivers import MolecularData
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import time
import numpy as np
from mindquantum import Simulator, Hamiltonian, X, Circuit
from scipy.optimize import minimize
#from mindquantum.algorithm import generate_uccsd
import matplotlib.pyplot as plt
from uccsd_generator import Uccsd_Generator

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
    def __init__(self, molecule=None, amp_th=0, seed=1202, file=None):
        #self.timer = Timer()
        self.molecule = molecule
        self.amp_th = amp_th
        self.backend = 'projectq'
        self.seed = seed
        self.file = file
        self.init_amp = []

        if molecule != None:
            self.generate_circuit(molecule)

    def generate_circuit(self, molecule=None, seed=1202):
        if molecule == None:
            molecule = self.molecule
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

        ansatz_circuit, \
        self.init_amp, \
        self.params_name, \
        self.hamiltonian, \
        self.n_qubits, \
        self.n_electrons = Uccsd_Generator(molecule, self.amp_th, 'JW').generate_uccsd

        self.circuit += ansatz_circuit 
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

    def optimize(self, operator=None, circuit=None, init_amp=[],
                 method='bfgs', maxstep=200, iter_info=False, tol=0.032):
        if operator == None:
            operator = self.hamiltonian
        if circuit == None:
            circuit = self.circuit
        if np.array(init_amp).size == 0:
            init_amp = self.init_amp

        grad_ops = self.simulator.get_expectation_with_grad(Hamiltonian(operator), circuit)
        self.res = minimize(func, init_amp,
                            args=(grad_ops, iter_info),
                            method=method,
                            jac=True, 
                            options={'gtol': tol}
                            )

class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    '''    
    def run(self, prefix, molecular_file):
        try:
            return self._run(prefix=prefix, molecular_file=molecular_file)
        except Exception as e:
            print(f'{prefix} throw an error: ', e)
            return 0
    '''
    def run(self, prefix, molecular_file):
        ath = 0.005
        #if 'lih' in prefix.lower() or 'lih' in molecular_file.lower():
        #    ath = 0.005
        if 'ch4' in prefix.lower() or 'ch4' in molecular_file.lower():
            ath = 0.001

        molecule = load_molecular_file(molecular_file)
        #molecule.load()

        vqe = VQEoptimizer(amp_th=ath)

        mol = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=0)
        vqe.generate_circuit(mol)

        vqe.optimize(tol=0.001)
        en = vqe.res.fun

        return en
