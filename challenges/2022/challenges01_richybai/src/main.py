import os
os.environ['OMP_NUM_THREADS'] = '4'
import sys
from openfermion import MolecularData
from openfermionpyscf import run_pyscf
import time
import numpy as np
from mindquantum import Simulator, Hamiltonian, X, Circuit, RY
from mindquantum import QubitUCCAnsatz, InteractionOperator, get_fermion_operator, Transform
from uccsd_generator import Uccsd_Generator
from scipy.optimize import minimize

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


def func(x, grad_ops, baseline, show_iter_val=False):
    f, g = grad_ops(x)
    # 达到精度, 返回0梯度, 结束训练
    if f < baseline + 0.0016:
        g = np.zeros_like(g) 
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
    # 计算fci作为能量下降的baseline
    if 'h6' in molecule.filename.lower():
        molecule = run_pyscf(molecule, run_fci=1)
    if 'lih' in molecule.filename.lower():
        molecule = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=1)
    return molecule


class VQEoptimizer:
    def __init__(self, seed=1202):
        
        self.backend = 'projectq'
        self.seed = seed

    def generate_circuit_and_hamiltonian(self, molecule):
        
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

        if "h6" in molecule.filename.lower():
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
            self.simulator = Simulator(self.backend, molecule.n_qubits, self.seed)

            ham_of = molecule.get_molecular_hamiltonian()
            inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
            ham_hiq = get_fermion_operator(inter_ops)
            qubit_hamiltonian = Transform(ham_hiq).jordan_wigner()
            qubit_hamiltonian.compress()
            qubit_hamiltonian = qubit_hamiltonian.real

            self.init_params = np.zeros(len(self.circuit.params_name))
        
        if "lih" in molecule.filename.lower():
            
            ansatz_circuit, \
            self.init_params, \
            self.params_name, \
            qubit_hamiltonian, \
            self.n_qubits, \
            self.n_electrons = Uccsd_Generator(molecule, 0.005, 'JW').generate_uccsd

            self.circuit += ansatz_circuit 
            self.simulator = Simulator(self.backend, molecule.n_qubits, self.seed)

        self.hamiltonian = Hamiltonian(qubit_hamiltonian)



    def optimize(self, baseline, method='bfgs', iter_info=False, tol=1e-5):

        grad_ops = self.simulator.get_expectation_with_grad(self.hamiltonian, self.circuit)
        self.res = minimize(func, self.init_params,
                            args=(grad_ops, baseline, iter_info),
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
    # eval 代码里只调用了这一个函数，这个函数的输入和返回不能变
    def run(self, prefix, molecular_file):
        """
        prefix: 函数的名称
        molecular_file: 分子结构存储的地方
        return 基态能量
        """
        # 加载分子数据
        molecule = load_molecular_file(molecular_file)
        vqe = VQEoptimizer()
        vqe.generate_circuit_and_hamiltonian(molecule)

        vqe.optimize(molecule.fci_energy, tol=1e-5)
        en = vqe.res.fun

        return en


# import os
# os.environ['OMP_NUM_THREADS'] = '4'
# import sys
# #from hiqfermion.drivers import MolecularData
# from openfermion import MolecularData
# from openfermionpyscf import run_pyscf
# import time
# import numpy as np
# from mindquantum import Simulator, Hamiltonian, X, Circuit
# from scipy.optimize import minimize
# #from mindquantum.algorithm import generate_uccsd
# import matplotlib.pyplot as plt
# from uccsd_generator import Uccsd_Generator

# class Timer:
#     def __init__(self, t0=0.0):
#         self.start_time = time.time()
#         self.t0 = t0

#     def runtime(self):
#         return time.time() - self.start_time + self.t0

#     def resetime(self):
#         self.start_time = time.time()

# def format_time(t):
#     hh = t // 3600
#     mm = (t - 3600 * hh) // 60
#     ss = t % 60
#     return hh, mm, ss


# def func(x, grad_ops, show_iter_val=False):
#     f, g = grad_ops(x)
#     if show_iter_val:
#         print(np.real(np.squeeze(f)))
#     return np.real(np.squeeze(f)), np.squeeze(g)

# def param2dict(keys, values):
#     param_dict = {}
#     for (key, value) in zip(keys, values):
#         param_dict[key] = value
#     return param_dict

# def load_molecular_file(molecular_file):
#     molecule = MolecularData(filename=molecular_file, data_directory='./src/hdf5files/')
#     molecule = MolecularData(geometry=molecule.geometry, 
#                             basis=molecule.basis, 
#                             multiplicity=molecule.multiplicity,  
#                             charge=molecule.charge, 
#                             filename=molecular_file,
#                             data_directory='./src/hdf5files/')
#     return molecule

# class VQEoptimizer:
#     def __init__(self, molecule=None, amp_th=0, seed=1202, file=None):
#         #self.timer = Timer()
#         self.molecule = molecule
#         self.amp_th = amp_th
#         self.backend = 'projectq'
#         self.seed = seed
#         self.file = file
#         self.init_amp = []

#         if molecule != None:
#             self.generate_circuit(molecule)

#     def generate_circuit(self, molecule=None, seed=1202):
#         if molecule == None:
#             molecule = self.molecule
#         self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

#         ansatz_circuit, \
#         self.init_amp, \
#         self.params_name, \
#         self.hamiltonian, \
#         self.n_qubits, \
#         self.n_electrons = Uccsd_Generator(molecule, self.amp_th, 'JW').generate_uccsd

#         self.circuit += ansatz_circuit 
#         self.simulator = Simulator(self.backend, self.n_qubits, seed)

#     def optimize(self, operator=None, circuit=None, init_amp=[],
#                  method='bfgs', maxstep=200, iter_info=False, tol=1e-5):
#         if operator == None:
#             operator = self.hamiltonian
#         if circuit == None:
#             circuit = self.circuit
#         if np.array(init_amp).size == 0:
#             init_amp = self.init_amp

#         grad_ops = self.simulator.get_expectation_with_grad(Hamiltonian(operator), circuit)
#         self.res = minimize(func, init_amp,
#                             args=(grad_ops, iter_info),
#                             method=method,
#                             jac=True, 
#                             options={'gtol': tol}
#                             )

# class Main:
#     def __init__(self):
#         super().__init__()
#         self.work_dir = './src/'

#     '''    
#     def run(self, prefix, molecular_file):
#         try:
#             return self._run(prefix=prefix, molecular_file=molecular_file)
#         except Exception as e:
#             print(f'{prefix} throw an error: ', e)
#             return 0
#     '''
#     def run(self, prefix, molecular_file):
#         ath = 0.005
#         #if 'lih' in prefix.lower() or 'lih' in molecular_file.lower():
#         #    ath = 0.005
#         if 'ch4' in prefix.lower() or 'ch4' in molecular_file.lower():
#             ath = 0.001

#         molecule = load_molecular_file(molecular_file)
#         #molecule.load()

#         vqe = VQEoptimizer(amp_th=ath)

#         mol = run_pyscf(molecule, run_scf=1, run_ccsd=1, run_fci=0)
#         vqe.generate_circuit(mol)

#         vqe.optimize(tol=1e-5)
#         en = vqe.res.fun

#         return en
