import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import time
import itertools
import numpy as np
import mindspore as ms

from hiqfermion.drivers import MolecularData
from openfermion import InteractionOperator
from openfermion.utils.indexing import down_index, up_index
from openfermionpyscf import run_pyscf
from mindquantum.core.operators.utils import hermitian_conjugated
from mindquantum import Simulator, Hamiltonian, X, Circuit, Transform, UCCAnsatz
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum import TimeEvolution
import mindspore.context as context

from mindquantum.core.operators import PolynomialTensor, get_fermion_operator
from mindquantum.core.parameterresolver import ParameterResolver as PR

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

###########################################################################

def get_intagral_from_tensor(one_body_tensor, two_body_tensor):
    n_qubits = one_body_tensor.shape[0]

    one_body_integral = np.zeros(shape=(n_qubits // 2, n_qubits // 2))
    two_body_integral = np.zeros(shape=(n_qubits // 2, n_qubits // 2, n_qubits // 2, n_qubits // 2))

    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
            one_body_integral[p, q] = one_body_tensor[2 * p, 2 * q]

            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    two_body_integral[p, q, r, s] = two_body_tensor[2 * p, 2 * q, 2 * r, 2 * s] * 2

    return one_body_integral, two_body_integral


def get_active_space_integral(one_body_integral,
                              two_body_integral,
                              occupied_indices,
                              active_indices):
    # Fix data type for a few edge cases
    occupied_indices = [] if occupied_indices is None else occupied_indices
    if (len(active_indices) < 1):
        raise ValueError('Some active indices required for reduction.')

    # Determine core constant
    constant_adujust = 0.0
    for i in occupied_indices:
        # This part come from one_body interaction, if we remove the spin orbital,
        # just add an energy shift for one_body hamiltonian.
        constant_adujust += 2 * one_body_integral[i, i]

        for j in occupied_indices:
            # This part come from the two_body interaction, where the first item is energy shift
            # the second item is because fermion operator anti-commute.
            constant_adujust += (2 * two_body_integral[i, j, j, i] - two_body_integral[i, j, i, j])

    # Modified one electron integrals
    one_body_integral_new = np.copy(one_body_integral)
    for u in active_indices:
        for v in active_indices:
            for i in occupied_indices:
                # If we remove some spin orbital, some Hpqrs would become Hpq, another would be energy shift.
                one_body_integral_new[u, v] += (2 * two_body_integral[i, u, v, i] - two_body_integral[i, u, i, v])

    return (constant_adujust,
            one_body_integral_new[np.ix_(active_indices, active_indices)],
            two_body_integral[np.ix_(active_indices, active_indices,
                                     active_indices, active_indices)])


def get_tensor_from_integral(one_body_integral, two_body_integral):
    n_qubits = 2 * one_body_integral.shape[0]

    one_body_tensor = np.zeros(shape=(n_qubits, n_qubits))
    two_body_tensor = np.zeros(shape=(n_qubits, n_qubits, n_qubits, n_qubits))

    for p in range(n_qubits // 2):
        for q in range(n_qubits // 2):
            one_body_tensor[2 * p, 2 * q] = one_body_integral[p, q]
            one_body_tensor[2 * p + 1, 2 * q + 1] = one_body_integral[p, q]

            for r in range(n_qubits // 2):
                for s in range(n_qubits // 2):
                    # Mixed spin
                    two_body_tensor[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (two_body_integral[p, q, r, s] / 2.)
                    two_body_tensor[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (two_body_integral[p, q, r, s] / 2.)

                    # Same spin
                    two_body_tensor[2 * p, 2 * q, 2 * r, 2 * s] = (two_body_integral[p, q, r, s] / 2.)
                    two_body_tensor[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = (two_body_integral[p, q, r, s] / 2.)

    return one_body_tensor, two_body_tensor


def remove_spin_orbital(ham_fermion, eigval_of_rdm, occupied_threshold, remove_threshold, check_orbital=False):
    electron_num = ham_fermion.n_qubits // 2
    if occupied_threshold <= remove_threshold:
        raise ValueError("The occupied threshold should lager then remove threshold")
    if len(eigval_of_rdm) != electron_num:
        raise ValueError("The length of eigval is not same as number of electron, please check the eigval")

    occupied_indices = []
    active_indices = []
    space_indices = [i for i in range(electron_num)]

    occupied_space_orbital = list(np.where(eigval_of_rdm >= occupied_threshold)[0])
    if len(occupied_space_orbital) > 0:
        occupied_indices = [i for i in occupied_space_orbital]

    remove_space_orbital = list(np.where(eigval_of_rdm <= remove_threshold)[0])
    if len(remove_space_orbital) >= 0:
        active_indices = set(space_indices) - (set(occupied_indices + remove_space_orbital))
        active_indices = [i for i in active_indices]

    # If the hamiltonian cannot be simplified, just return the original hamiltonian.
    if len(occupied_space_orbital) == 0 and len(remove_space_orbital) == 0:
        return ham_fermion, ([], [i for i in range(electron_num)])

    constant = ham_fermion.constant
    one_body_integral, two_body_integral = get_intagral_from_tensor(ham_fermion.one_body_tensor,
                                                                    ham_fermion.two_body_tensor)
    constant_adjustment, one_body_integral_new, two_body_integral_new = \
        get_active_space_integral(one_body_integral,
                                  two_body_integral,
                                  occupied_indices,
                                  active_indices)
    constant_new = constant + constant_adjustment
    one_body_tensor, two_body_tensor = get_tensor_from_integral(one_body_integral_new, two_body_integral_new)
    ham_fermion_new = InteractionOperator(constant_new, one_body_tensor, two_body_tensor)
    if check_orbital:
        return ham_fermion_new, (occupied_indices, active_indices)
    else:
        return ham_fermion_new


################################################################
def ucc_pair_excitation_circuit(occ_orb, vir_orb):
    occ_indices = []
    vir_indices = []
    n_occ = 0
    n_vir = 0
    if len(set(occ_orb)) != len(occ_orb):
        raise ValueError("Indices for occupied orbitals should be unique!")
    n_occ = len(occ_orb)
    occ_indices = occ_orb

    if len(set(vir_orb)) != len(vir_orb):
        raise ValueError("Indices for virtual orbitals should be unique!")
    n_vir = len(vir_orb)
    vir_indices = vir_orb

    if set(occ_indices).intersection(vir_indices):
        raise ValueError("Occupied and virtual orbitals should be different!")

    from mindquantum.core.operators import FermionOperator
    # Initialize operator
    op = FermionOperator()

    # Generate pair excitations
    for pq_counter, (p_idx, q_idx) in enumerate(
            itertools.product(range(n_vir), range(n_occ))):
        # Get indices of spatial orbitals
        p = vir_indices[p_idx]
        q = occ_indices[q_idx]

        p_up = up_index(p)
        p_down = down_index(p)
        q_up = up_index(q)
        q_down = down_index(q)
        # Generate excitation
        coeff = PR({f'd_{pq_counter}': 1})
        tppqq = FermionOperator(((p_up, 1), (q_up, 0), (p_down, 1), (q_down, 0)), coeff)
        op += tppqq - hermitian_conjugated(tppqq)

    circuit = TimeEvolution(Transform(op).jordan_wigner().imag, 1).circuit

    return circuit




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


def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


class VQEoptimizer:
    def __init__(self, max_threshold, min_threshold, molecule=None, seed=42, file=None):
        self.timer = Timer()
        self.molecule = molecule
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.backend = 'projectq'
        self.seed = seed
        self.file = file

        if molecule != None:
            self.remove_orbital(molecule, self.max_threshold, self.min_threshold)
            self.generate_circuit(molecule)

        print("Initialize finished! Time: %.2f s" % self.timer.runtime(), file=self.file)
        sys.stdout.flush()
    
    def remove_orbital(self, max_threshold, min_threshold, molecule=None):
        if molecule == None:
            molecule = self.molecule

        self.ham_fermion = molecule.get_molecular_hamiltonian()
        # Rotation spin orbital
        odm_fci = molecule.fci_one_rdm
        self.occupied_num_MO, eigvec = np.linalg.eig(odm_fci)
        Umatrix = np.kron(eigvec, np.eye(2))
        self.ham_fermion.rotate_basis(Umatrix)

        # Remove molecular orbital
        self.remove_ham, self.indices = remove_spin_orbital(self.ham_fermion, self.occupied_num_MO, max_threshold,
                                                            min_threshold, True)

        # Get the fermion hamiltonian after rotation in openfermion
        new_constant = self.remove_ham.constant
        new_one_body_tensor = self.remove_ham.one_body_tensor
        new_two_body_tensor = self.remove_ham.two_body_tensor

        # Reconstruction fermion operator in mindquantum
        poly_operator = PolynomialTensor({(): new_constant,
                                          (1, 0): new_one_body_tensor,
                                          (1, 1, 0, 0): new_two_body_tensor})
        fermion_operator = get_fermion_operator(poly_operator)

        # Get the pauli operator after jordan wigner
        self.ham_pauli = Transform(fermion_operator).jordan_wigner()
    
    
    def generate_circuit(self, ansatz_type, molecule=None):
        if molecule == None:
            molecule = self.molecule
        occupied_mo_new = self.occupied_num_MO[self.indices[1]]
        occ_orb = np.where(occupied_mo_new > 1.0)[0]
        vir_orb = np.delete(np.arange(len(occupied_mo_new)), occ_orb)

        high_occupied_spin_indice = np.array([[2 * i, 2 * i + 1] for i in occ_orb]).reshape([-1])
        high_occupied_spin_indice = [int(i) for i in high_occupied_spin_indice]

        self.circuit = Circuit([X.on(i) for i in high_occupied_spin_indice])
        self.n_qubits = len(occupied_mo_new) * 2

        if ansatz_type == "UCCPE":
            ansatz_circuit = ucc_pair_excitation_circuit([int(i) for i in occ_orb], [int(i) for i in vir_orb])
        else:
            ansatz_circuit = UCCAnsatz(self.n_qubits,
                                       molecule.n_electrons - len(self.indices[0]) * 2,
                                       [int(i) for i in occ_orb],
                                       [int(i) for i in vir_orb], trotter_step=1).circuit
        self.circuit += ansatz_circuit

    def optimize(self, learning_rate, eps, file, operator=None, circuit=None):
        if operator == None:
            operator = self.ham_pauli
        if circuit == None:
            circuit = self.circuit

        self.simulator = Simulator(self.backend, circuit.n_qubits)

        molecule_pqc = self.simulator.get_expectation_with_grad(Hamiltonian(operator.real), circuit)
        molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc, 'Zeros')

        initial_energy = molecule_pqcnet()

        optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=learning_rate)
        train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)
        energy_diff = eps * 1000
        energy_last = initial_energy.asnumpy() + energy_diff
        iter_idx = 0
        while abs(energy_diff) > eps:
            self.energy_i = train_pqcnet().asnumpy()
            energy_diff = energy_last - self.energy_i
            energy_last = self.energy_i
            iter_idx += 1

        self.optimal_params = molecule_pqcnet.weight.asnumpy()
        print("The iterations is: {}, and the final convergent energy is: {}".format(iter_idx, self.energy_i),
              file=file)
    

class Main:
    def __init__(self):
        super().__init__()
        self.work_dir = os.path.dirname(os.path.abspath(__file__))

    def run(self, prefix, molecular_file):
        prefix = prefix
        molecule = MolecularData(filename=molecular_file, data_directory='./src/hdf5files/')
        molecule.load()

        with open(self.work_dir + "/" + prefix + '.o', 'a') as f:
            print(f)
            print('Start case: ', prefix, file=f)
            print('OMP_NUM_THREAD: ', os.environ['OMP_NUM_THREADS'], file=f)
            max_threshold = 1.9995
            min_threshold = 8e-4
            vqe = VQEoptimizer(max_threshold, min_threshold, file=f)
            en_list, time_list = [], []

            
            molecule = MolecularData(geometry=molecule.geometry, 
                                    basis=molecule.basis, 
                                    multiplicity=molecule.multiplicity,  
                                    charge=molecule.charge, 
                                    filename=molecular_file,
                                    data_directory='./src/hdf5files/')
            

            mol = run_pyscf(molecule, run_fci=1)
            vqe.remove_orbital(max_threshold, min_threshold, mol)
            
            if prefix == "LiH":
                
                vqe.generate_circuit("UCCPE", mol)
                vqe.optimize(learning_rate=0.25, eps=1e-4, file=f)
                
                #    vqe.generate_circuit("UCCAnsatz", mol)
                #    vqe.optimize(learning_rate=0.08, eps=1e-4, file=f)
            else:
                vqe.generate_circuit("UCCAnsatz", mol)
                vqe.optimize(learning_rate=0.066, eps=5e-4, file=f)
            
            param_dict = param2dict(vqe.circuit.params_name, vqe.optimal_params)

            vqe.simulator.reset()
            vqe.simulator.apply_circuit(vqe.circuit, param_dict)
            t = vqe.timer.runtime()
            en = vqe.simulator.get_expectation(Hamiltonian(vqe.ham_pauli.real)).real
            print('Time: %i hrs %i mints %.2f sec.' % format_time(t), 'Energy: ', en, file=f)
            sys.stdout.flush()
            
            en_list.append(en)
            time_list.append(t)

            print('Optimization completed. Time: %i hrs %i mints %.2f sec.' % format_time(vqe.timer.runtime()), file=f)

        return en_list[0]
        

