import os

os.environ['OMP_NUM_THREADS'] = '4'
import sys
from openfermionpyscf import run_pyscf
import time
import numpy as np
from mindquantum import Simulator, Hamiltonian, X, Circuit, CNOTGate, Z, RY, BarrierGate
from scipy.optimize import minimize
from mindquantum.algorithm.nisq.chem.uccsd import _para_uccsd_singlet_generator
import matplotlib.pyplot as plt
#from hiqfermion.drivers import MolecularData
from openfermion.chem import MolecularData
from mindquantum.core.operators.utils import get_fermion_operator
from mindquantum.algorithm.nisq.chem.transform import Transform
from mindquantum.third_party.interaction_operator import InteractionOperator
#from mindquantum.core.circuit import Circuit


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
        #print(np.squeeze(f), file=file)
        sys.stdout.flush()
    return np.real(np.squeeze(f)), np.squeeze(g)


def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


def processing_fermion_ansatz(fermion_ansatz, parameters_name):
    fadict = {}
    for i in fermion_ansatz:
        b = 1
        if not i[1] in parameters_name:
            continue
        fermion_operator = i[0]
        ol = list(fermion_operator.terms.keys())[0]
        ol = [j[0] for j in ol]
        if len(ol) > 2:
            if ol[1] > ol[3]:
                b *= 1
            elif ol[1] < ol[3]:
                b *= -1
            else:
                b *= 0
            if ol[0] > ol[2]:
                b *= 1
            elif ol[0] < ol[2]:
                b *= -1
            else:
                b *= 0
        if b == 0:
            continue
        ol = tuple(sorted(ol))
        if not (ol, i[1]) in fadict:
            fadict[(ol, i[1])] = b
        else:
            fadict[(ol, i[1])] += b
    return fadict


def efficientCircuit2(fermion_ansatz, parameters_name):
    fadict = processing_fermion_ansatz(fermion_ansatz, parameters_name)
    circ = Circuit()
    x = 0
    for i in fadict:
        ilist = i[0]
        p = i[1]
        l = len(ilist)
        ccc = Circuit()
        if l == 2:
            for m in range(ilist[1] - 2, ilist[0], -1):
                ccc += CNOTGate().on(m, m + 1)
            ccc += CNOTGate().on(ilist[0], ilist[1])
            if ilist[0] + 1 != ilist[1]:
                ccc += Z(ilist[1], ilist[0] + 1)
            ccc += RY({p: 2 * fadict[i]}).on(ilist[1], ilist[0])
            if ilist[0] + 1 != ilist[1]:
                ccc += Z(ilist[1], ilist[0] + 1)
            ccc += CNOTGate().on(ilist[0], ilist[1])
            for m in range(ilist[0] + 1, ilist[1] - 1):
                ccc += CNOTGate().on(m, m + 1)

        else:
            lll = set(range(ilist[0] + 1, ilist[1])) | set(
                range(ilist[2] + 1, ilist[3]))
            lll = sorted(list(lll))[::-1]

            for m in lll[1:]:
                if m == ilist[1] - 1:
                    ccc += CNOTGate().on(m, ilist[2] + 1)
                else:
                    ccc += CNOTGate().on(m, m + 1)
            ccc += CNOTGate().on(ilist[0], ilist[1])
            ccc += CNOTGate().on(ilist[2], ilist[3])
            ccc += CNOTGate().on(ilist[1], ilist[3])
            if lll != []:
                ccc += Z(ilist[3], lll[-1])
            ccc = ccc + X(ilist[0]) + X(ilist[2])
            ccc += RY({p: -2 * fadict[i]}).on(ilist[3], ilist[:3])
            ccc = ccc + X(ilist[0]) + X(ilist[2])
            if lll != []:
                ccc += Z(ilist[3], lll[-1])
            ccc += CNOTGate().on(ilist[1], ilist[3])
            ccc += CNOTGate().on(ilist[2], ilist[3])
            ccc += CNOTGate().on(ilist[0], ilist[1])
            for m in lll[1:][::-1]:
                if m == ilist[1] - 1:
                    ccc += CNOTGate().on(m, ilist[2] + 1)
                else:
                    ccc += CNOTGate().on(m, m + 1)
        ccc += BarrierGate()
        circ += ccc
    return circ


# def score(pauli_ansatz, qubit_hamiltonian):
#     '''给每个参数的重要性打分'''
#     scores = dict()
#     qh = qubit_hamiltonian.terms
#     for ansatz_pauli_string in pauli_ansatz:
#         parameter = pauli_ansatz[ansatz_pauli_string]
#         parameter = list(parameter.keys())[0]
#         if not parameter in scores:
#             scores[parameter]=0
#         for hamilt_pauli_string in qh:
#             d = 0
#             for i in ansatz_pauli_string:
#                 for j in hamilt_pauli_string:
#                     if i[0]==j[0] and i[1]!=j[1]:
#                         d+=1
#                         break
#             scores[parameter] += 2 ** d * np.abs(qh[hamilt_pauli_string])
#     return scores


def generate_efficient_uccsd(molecular, th=0.001):
    """
    Generate a uccsd quantum circuit based on a molecular data generated by
    HiQfermion or openfermion.

    Args:
        molecular (Union[str, MolecularData]): the name of the molecular data file,
            or openfermion MolecularData.
        th (int): the threshold to filt the uccsd amplitude. When th < 0, we
            will keep all amplitudes. When th == 0, we will keep all amplitude
            that are positive. Default: 0.

    Returns:
        - **uccsd_circuit** (Circuit), the ansatz circuit generated by uccsd method.
        - **initial_amplitudes** (numpy.ndarray), the initial parameter values of uccsd circuit.
        - **parameters_name** (list[str]), the name of initial parameters.
        - **qubit_hamiltonian** (QubitOperator), the hamiltonian of the molecule.
        - **n_qubits** (int), the number of qubits in simulation.
        - **n_electrons**, the number of electrons of the molecule.
    """
    if isinstance(molecular, str):
        mol = MolecularData(filename=molecular)
        mol.load()
    else:
        mol = molecular
    #print("ccsd:{}.".format(mol.ccsd_energy))
    #print("fci:{}.".format(mol.fci_energy))
    fermion_ansatz, parameters = _para_uccsd_singlet_generator(mol, th)
    print(len(parameters))
    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)
    qubit_hamiltonian = Transform(ham_hiq).jordan_wigner()
    qubit_hamiltonian.compress()
    parameters_name = list(parameters.keys())
    #print(parameters_name)
    initial_amplitudes = [parameters[i] for i in parameters_name]
    #print(initial_amplitudes)
    #uccsd_circuit = efficientCircuit(pauli_ansatz, parameters_name)
    uccsd_circuit = efficientCircuit2(fermion_ansatz, parameters_name)
    return uccsd_circuit, \
        initial_amplitudes, \
        parameters_name, \
        qubit_hamiltonian, \
        mol.n_qubits, \
        mol.n_electrons


class VQEoptimizer:
    def __init__(self, molecule=None, seed=1202, file=None):
        self.timer = Timer()
        self.molecule = molecule
        self.backend = 'mqvector'
        self.seed = seed
        self.file = file
        self.init_amp = []

        if molecule != None:
            self.generate_circuit(molecule)

        #print("Initialize finished! Time: %.2f s" % self.timer.runtime(), file=self.file)
        sys.stdout.flush()

    def generate_circuit(self, molecule=None, seed=1202, th=0.001):
        if molecule == None:
            molecule = self.molecule
        self.circuit = Circuit([X.on(i) for i in range(molecule.n_electrons)])

        ansatz_circuit, \
        self.init_amp, \
        self.params_name, \
        self.hamiltonian, \
        self.n_qubits, \
        self.n_electrons = generate_efficient_uccsd(molecule, th)

        self.circuit += ansatz_circuit
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

    def optimize(self,
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
        grad_ops = self.simulator.get_expectation_with_grad(
            Hamiltonian(operator), circuit)
        self.res = minimize(func,
                            init_amp,
                            args=(grad_ops, self.file, iter_info),
                            method=method,
                            jac=True,
                            options={'disp': True})
        #print(self.res)


class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = './src/'

    def run(self, prefix, molecular_file, geom_list):
        if prefix == 'LiH':
            th = [
                0.008, 0.02, 0.015, 0.011, 0.011, 0.055, 0.012, 0.06, 0.06,
                0.055
            ]
        elif prefix == 'CH4':
            th = [0.001, 0.001]
        else:
            th = [0.001] * len(geom_list)
        prefix = prefix
        molecule = MolecularData(filename=self.work_dir + molecular_file)
        molecule.load()
        x = []
        with open(self.work_dir + prefix + '.o', 'a') as f:
            print(f)
            print('Start case: ', prefix, file=f)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'], file=f)
            vqe = VQEoptimizer(file=f)
            en_list, time_list = [], []
            fcienergy = []
            for i in range(len(geom_list)):
                mol0 = MolecularData(geometry=geom_list[i],
                                     basis=molecule.basis,
                                     charge=molecule.charge,
                                     multiplicity=molecule.multiplicity,
                                     data_directory='./')
                mol = run_pyscf(mol0, run_scf=0, run_ccsd=1, run_fci=1)
                #vqe.generate_circuit(mol, th=th[i])
                # fcienergy.append(mol.fci_energy)
                # vqe.optimize()
                # param_dict = param2dict(vqe.circuit.params_name, vqe.res.x)
                #
                # vqe.simulator.apply_circuit(vqe.circuit, param_dict)
                # t = vqe.timer.runtime()
                # en = vqe.simulator.get_expectation(Hamiltonian(vqe.hamiltonian)).real
                #print('en:', en)
                print('fci_energy', mol.fci_energy)
                print('Time: %i hrs %i mints %.2f sec.' % format_time(t),
                      'Energy: ',
                      en,
                      file=f)
                sys.stdout.flush()
                en_list.append(en)
                time_list.append(t)

        print('fcienergy', fcienergy)

        #print('Optimization completed. Time: %i hrs %i mints %.2f sec.' % format_time(vqe.timer.runtime()),file=f)
        return en_list, time_list  # , nparam_list


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
