import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys
from hiqfermion.drivers import MolecularData
# from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.algorithm.nisq.chem import get_qubit_hamiltonian
import time
import numpy as np
from mindquantum import Simulator, Hamiltonian, RY, H, X, Circuit
from scipy.optimize import minimize
# from mindquantum.algorithm import generate_uccsd
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


def get_possible_excitations(electrons, orbitals, delta_sz=0):
    """
    **Example**
    -> electrons = 2
    -> orbitals = 4
    -> singles, doubles = excitations(electrons, orbitals)
    -> print(singles)
    [[0, 2], [1, 3]]
    - print(doubles)
    [[0, 1, 2, 3]]
    """
    if not electrons > 0:
        raise ValueError(
            f"The number of active electrons has to be greater than 0 \n"
            f"Got n_electrons = {electrons}")

    if orbitals <= electrons:
        raise ValueError(
            f"The number of active spin-orbitals ({orbitals}) "
            f"has to be greater than the number of active electrons ({electrons})."
        )

    if delta_sz not in (0, 1, -1, 2, -2):
        raise ValueError(
            f"Expected values for 'delta_sz' are 0, +/- 1 and +/- 2 but got ({delta_sz})."
        )

    # define the spin projection 'sz' of the single-particle states
    sz = np.array([0.5 if (i % 2 == 0) else -0.5 for i in range(orbitals)])

    singles = [[r, p] for r in range(electrons)
               for p in range(electrons, orbitals)
               if sz[p] - sz[r] == delta_sz]

    doubles = [[s, r, q, p] for s in range(electrons - 1)
               for r in range(s + 1, electrons)
               for q in range(electrons, orbitals - 1)
               for p in range(q + 1, orbitals)
               if (sz[p] + sz[q] - sz[r] - sz[s]) == delta_sz]

    return singles, doubles


def single_excitation(i, qubits):
    circ = Circuit()
    circ += X.on(qubits[1], qubits[0])
    circ += RY(f's{i}').on(qubits[0], qubits[1])
    circ += X.on(qubits[1], qubits[0])
    return circ


def double_excitation(i, qubits):
    circ = Circuit()
    circ += X.on(qubits[3], qubits[2])
    circ += X.on(qubits[2], qubits[0])
    circ += H.on(qubits[3])
    circ += H.on(qubits[0])
    circ += X.on(qubits[3], qubits[2])
    circ += X.on(qubits[1], qubits[0])
    circ += RY({f'd{i}': 1 / 8}).on(qubits[1])
    circ += RY({f'd{i}': 1 / 8}).on(qubits[0])
    circ += X.on(qubits[3], qubits[0])
    circ += H.on(qubits[3])
    circ += X.on(qubits[1], qubits[3])
    circ += RY({f'd{i}': 1 / 8}).on(qubits[1])
    circ += RY({f'd{i}': -1 / 8}).on(qubits[0])
    circ += X.on(qubits[1], qubits[2])
    circ += X.on(qubits[0], qubits[2])
    circ += RY({f'd{i}': -1 / 8}).on(qubits[1])
    circ += RY({f'd{i}': 1 / 8}).on(qubits[0])
    circ += X.on(qubits[1], qubits[3])
    circ += H.on(qubits[3])
    circ += X.on(qubits[3], qubits[0])
    circ += RY({f'd{i}': -1 / 8}).on(qubits[1])
    circ += RY({f'd{i}': 1 / 8}).on(qubits[0])
    circ += X.on(qubits[1], qubits[0])
    circ += X.on(qubits[0], qubits[2])
    circ += H.on(qubits[0])
    circ += H.on(qubits[3])
    circ += X.on(qubits[2], qubits[0])
    circ += X.on(qubits[3], qubits[2])
    return circ


class VQEoptimizer:
    def __init__(self, molecule=None, seed=1202, file=None):
        self.timer = Timer()
        self.molecule = molecule
        #         self.init_amp = []
        #         self.params_name = None
        #         self.hamiltonian = None
        #         self.n_qubits = 0
        self.qubits_index = None
        #         self.n_electrons = 0
        #         self.double_excitations = []
        #         self.single_excitations = []
        #         self.circuit = None
        self.backend = 'mqvector'
        self.seed = seed
        self.file = file

        if molecule != None:
            self.n_electrons = molecule.n_electrons
            self.n_qubits = molecule.n_qubits
            self.qubits_index = list(range(molecule.n_qubits))
            self.hamiltonian = Hamiltonian(get_qubit_hamiltonian(molecule))

        print("Initialize finished! Time: %.2f s" % self.timer.runtime(),
              file=self.file)
        sys.stdout.flush()

    def select_valid_excitations(self):
        singles, doubles = get_possible_excitations(self.n_electrons,
                                                    self.n_qubits,
                                                    delta_sz=0)

        #select_valid_doubles
        sel_doubles_circuit = self.circuit_1(doubles)
        grad_ops = self.simulator.get_expectation_with_grad(
            self.hamiltonian, sel_doubles_circuit)
        init_params = [0] * len(doubles)
        ed, gd = grad_ops(np.array(init_params))
        gd = gd[0][0].real
        self.double_excitations = [
            doubles[i] for i in range(len(doubles)) if abs(gd[i]) > 1.0e-5
        ]
        #update_doubles_parameters, check len(self.double_excitations) > 0
        op_doubles_circuit = self.circuit_1(self.double_excitations)
        grad_ops = self.simulator.get_expectation_with_grad(
            self.hamiltonian, op_doubles_circuit)
        params_d = [0] * len(self.double_excitations)  #init double parameters

        res = minimize(func,
                       params_d,
                       args=(grad_ops, self.file),
                       method='bfgs',
                       jac=True)
        params_d = res.x.real  #update double parameters

        select_singles_circuit1, select_singles_circuit2 = self.circuit_2(
            singles, self.double_excitations)
        select_singles_circuit1 = select_singles_circuit1.no_grad()
        ops_singles_circuit = select_singles_circuit1.as_encoder(
        ) + select_singles_circuit2

        grad_ops = self.simulator.get_expectation_with_grad(
            self.hamiltonian,
            ops_singles_circuit,
        )
        params_s = [0] * len(singles)
        e, ge, gs = grad_ops(np.array([params_d]), np.array(params_s))
        gs = gs[0][0].real
        self.single_excitations = [
            singles[i] for i in range(len(singles)) if abs(gs[i]) > 1.0e-5
        ]

    def generate_circuit(self, molecule=None, seed=1202):
        if molecule == None:
            molecule = self.molecule
            self.n_electrons = molecule.n_electrons
            self.n_qubits = molecule.n_qubits
            self.qubits_index = list(range(molecule.n_qubits))
            self.hamiltonian = Hamiltonian(get_qubit_hamiltonian(molecule))
        else:
            self.molecule = molecule
            self.n_electrons = molecule.n_electrons
            self.n_qubits = molecule.n_qubits
            self.qubits_index = list(range(molecule.n_qubits))
            self.hamiltonian = Hamiltonian(get_qubit_hamiltonian(molecule))
        self.simulator = Simulator(self.backend, self.n_qubits, seed)

        self.select_valid_excitations()
        optimal_excitations = self.double_excitations + self.single_excitations
        self.circuit = self.circuit_1(optimal_excitations)

    def circuit_1(self, excitations):
        circ = Circuit()
        circ += Circuit([X.on(i) for i in range(self.n_electrons)])  # hf state

        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                circ += double_excitation(i, excitation)
            else:
                circ += single_excitation(i, excitation)

        return circ

    def circuit_2(self, excitations, excitation_select):
        circ1 = Circuit()
        circ1 += Circuit([X.on(i)
                          for i in range(self.n_electrons)])  # hf state

        for i, excitation in enumerate(excitation_select):
            if len(excitation) == 4:
                circ1 += double_excitation(i, excitation)
            elif len(excitation) == 2:
                circ1 += single_excitation(i, excitation)
        circ2 = Circuit()
        for i, excitation in enumerate(excitations):
            if len(excitation) == 4:
                circ2 += double_excitation(i, excitation)
            elif len(excitation) == 2:
                circ2 += single_excitation(i, excitation)

        return circ1, circ2

    def optimize(self,
                 operator=None,
                 init_amp=[],
                 method='bfgs',
                 maxstep=200,
                 iter_info=False):
        if operator == None:
            operator = self.hamiltonian
        if np.array(init_amp).size == 0:
            init_amp = [0] * len(self.circuit.params_name)

        grad_ops = self.simulator.get_expectation_with_grad(
            operator, self.circuit)
        self.res = minimize(func,
                            init_amp,
                            args=(grad_ops, self.file, iter_info),
                            method=method,
                            jac=True)


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
                mol = run_pyscf(mol0, run_scf=1, run_ccsd=1, run_fci=0)

                vqe.generate_circuit(mol)
                vqe.optimize()
                param_dict = param2dict(vqe.circuit.params_name, vqe.res.x)

                vqe.simulator.apply_circuit(vqe.circuit, param_dict)
                t = vqe.timer.runtime()
                en = vqe.simulator.get_expectation(vqe.hamiltonian).real
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
