


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# type:ignore
import warnings
from enum import Enum, auto
from importlib.metadata import version
from itertools import combinations, product
from math import comb
# from typing import Mapping, Optional, Sequence
from mindquantum.core.circuit import Circuit as Circuit_ms
import numpy as np
# from braket.aws import AwsDevice
from openfermion.ops import FermionOperator, InteractionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from pyscf import ao2mo
from quri_parts.algo.ansatz import HardwareEfficient, SymmetryPreserving
from quri_parts.algo.optimizer import Adam, OptimizerStatus
# from quri_parts.braket.backend import BraketSamplingBackend
# from quri_parts.chem.ansatz import (
#     AllSinglesDoubles,
#     GateFabric,
#     ParticleConservingU1,
#     ParticleConservingU2,
# )
from mindquantum.simulator import Simulator
from quri_parts.chem.utils.spin import occupation_state_sz
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.core.estimator import (
    ConcurrentParametricQuantumEstimator,
    ConcurrentQuantumEstimator,
    Estimatable,
    Estimate,
)
from mindquantum.core.operators import FermionOperator as FermionOperator_ms
from mindquantum.core.operators import Hamiltonian as Hamiltonian_ms
from mindquantum.core.operators import QubitOperator as QubitOperator_ms
from mindquantum.algorithm.nisq import Transform as Transform_ms
from quri_parts.core.estimator.gradient import (
    create_parameter_shift_gradient_estimator,
)
# from quri_parts.core.estimator.sampling import create_sampling_estimator
# from quri_parts.core.measurement import bitwise_commuting_pauli_measurement
# from quri_parts.core.sampling import (
#     create_concurrent_sampler_from_sampling_backend,
# )
# from quri_parts.core.sampling.shots_allocator import (
#     create_weighted_random_shots_allocator,
# )
from quri_parts.core.state import (
    CircuitQuantumState,
    ParametricCircuitQuantumState,
)

from mindquantum.simulator.mqchem import MQChemSimulator, CIHamiltonian
from quri_parts.openfermion.ansatz import KUpCCGSD, TrotterUCCSD
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapping,
    jordan_wigner,
)
# from quri_parts.qiskit.backend import QiskitSamplingBackend
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)

from chemqulacs_namd.vqe.rdm_mindspore import get_1rdm, get_2rdm


from chemqulacs_namd.vqe.quri_to_mindquantum_circuit import quri_to_mindquantum


from scipy.optimize import minimize

class Backend:
    """Base class represents backend"""

    pass


class QulacsBackend(Backend):
    """Backend class for Qulacs"""

    pass




class Ansatz(Enum):
    """An enum representing an ansatz for VQE"""

    # HardwareEfficient = auto()
    # SymmetryPreserving = auto()
    # AllSinglesDoubles = auto()
    # ParticleConservingU1 = auto()
    # ParticleConservingU2 = auto()
    # GateFabric = auto()
    UCCSD = auto()
    KUpCCGSD = auto()
















def _get_active_hamiltonian(h1, h2, norb, ecore):
    n_orbitals = norb
    n_spin_orbitals = 2 * n_orbitals

    # Initialize Hamiltonian coefficients.
    one_body_coefficients = np.zeros((n_spin_orbitals, n_spin_orbitals))
    two_body_coefficients = np.zeros(
        (n_spin_orbitals, n_spin_orbitals, n_spin_orbitals, n_spin_orbitals)
    )
    # Set MO one and two electron-integrals
    # according to OpenFermion conventions
    one_body_integrals = h1
    h2_ = ao2mo.restore(
        1, h2.copy(), n_orbitals
    )  # no permutation see two_body_integrals of _pyscf_molecular_data.py
    two_body_integrals = np.asarray(h2_.transpose(0, 2, 3, 1), order="C")

    # Taken from OpenFermion
    # Loop through integrals.
    for p in range(n_spin_orbitals // 2):
        for q in range(n_spin_orbitals // 2):

            # Populate 1-body coefficients. Require p and q have same spin.
            one_body_coefficients[2 * p, 2 * q] = one_body_integrals[p, q]
            one_body_coefficients[2 * p + 1, 2 * q + 1] = one_body_integrals[p, q]
            # Continue looping to prepare 2-body coefficients.
            for r in range(n_spin_orbitals // 2):
                for s in range(n_spin_orbitals // 2):

                    # Mixed spin
                    two_body_coefficients[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    two_body_coefficients[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )

                    # Same spin
                    two_body_coefficients[2 * p, 2 * q, 2 * r, 2 * s] = (
                        two_body_integrals[p, q, r, s] / 2.0
                    )
                    two_body_coefficients[
                        2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1
                    ] = (two_body_integrals[p, q, r, s] / 2.0)

    # Get Hamiltonian in OpenFermion format
    active_hamiltonian = InteractionOperator(
        ecore, one_body_coefficients, two_body_coefficients
    )
    return active_hamiltonian


def _create_concurrent_estimators(
    backend: Backend, shots_per_iter: int
) -> tuple[
    ConcurrentQuantumEstimator[CircuitQuantumState],
    ConcurrentParametricQuantumEstimator[ParametricCircuitQuantumState],
]:
    if isinstance(backend, QulacsBackend):
        estimator = create_qulacs_vector_concurrent_estimator()
        parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
        return estimator, parametric_estimator
    else:
        estimator = create_qulacs_vector_concurrent_estimator()
        parametric_estimator = create_qulacs_vector_concurrent_parametric_estimator()
        return estimator, parametric_estimator



def _create_ansatz(
    ansatz: Ansatz,
    fermion_qubit_mapping: OpenFermionQubitMapping,
    n_sorbs: int,
    n_electrons: int,
    layers: int,
    k: int,
    trotter_number: int,
    include_pi: bool,
    use_singles: bool,
    delta_sz: int,
    singlet_excitation: bool,
) -> LinearMappedUnboundParametricQuantumCircuit:
    n_qubits = fermion_qubit_mapping.n_qubits_required(n_sorbs)
    # if ansatz == Ansatz.HardwareEfficient:
    #     return HardwareEfficient(n_qubits, layers)
    # elif ansatz == Ansatz.SymmetryPreserving:
    #     return SymmetryPreserving(n_qubits, layers)
    # elif ansatz == Ansatz.AllSinglesDoubles:
    #     return AllSinglesDoubles(n_qubits, n_electrons)
    # elif ansatz == Ansatz.ParticleConservingU1:
    #     return ParticleConservingU1(n_qubits, layers)
    # elif ansatz == Ansatz.ParticleConservingU2:
    #     return ParticleConservingU2(n_qubits, layers)
    # elif ansatz == Ansatz.GateFabric:
    #     return GateFabric(n_qubits, layers, include_pi)
    if ansatz == Ansatz.UCCSD:
        return TrotterUCCSD(
            n_sorbs,
            n_electrons,
            fermion_qubit_mapping,
            trotter_number,
            use_singles,
            delta_sz,
            singlet_excitation,
        )
    elif ansatz == Ansatz.KUpCCGSD:
        if version("quri-parts-openfermion") >= "0.19.0":
            return KUpCCGSD(
                n_sorbs,
                # n_electrons,
                k,
                fermion_qubit_mapping,
                trotter_number,
                delta_sz,
                singlet_excitation,
            )
        else:
            return KUpCCGSD(
                n_sorbs,
                n_electrons,
                k,
                fermion_qubit_mapping, 
                trotter_number,
                delta_sz,
                singlet_excitation,
            )


def vqe(init_params, cost_fn, grad_fn, optimizer):
    opt_state = optimizer.get_init_state(init_params)
    while True:
        opt_state = optimizer.step(opt_state, cost_fn, grad_fn)
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state


def generate_initial_states(
    n_orbitals, n_electron, excitation_number=0, fermion_qubit_mapping=jordan_wigner
):

    warnings.warn(
        "The function generates initial states for VQE and SSVQE"
        "If SSVQE is performed generate_initial_states only performs correctly for fermion_qubit_mapping=jordan_wigner. "
    )

    for m in range(n_electron, 2 * n_electron + 1):
        if comb(m, n_electron) >= excitation_number + 1:
            break
    else:
        raise Exception("excitation_number is too large")

    occ_indices_lst = sorted(
        list(combinations(range(m), n_electron)),
        key=lambda lst: sum([2**a for a in lst]),
    )[: excitation_number + 1]

    initial_states = []
    for occupied_indices in occ_indices_lst:
        sz = occupation_state_sz(occupied_indices)

        if version("quri-parts-openfermion") >= "0.19.0":
            state_mapper = fermion_qubit_mapping.get_state_mapper(
                n_spin_orbitals=2 * n_orbitals, n_fermions=n_electron, sz=sz
            )
        else:
            warnings.warn("SSVQE with Quri_parts < 0.19.0 is not tested.")
            state_mapper = fermion_qubit_mapping.get_state_mapper(
                n_spin_orbitals=2 * n_orbitals, n_fermions=n_electron
            )
        initial_states.append(state_mapper(occupied_indices))
    return initial_states


# ======================
class VQECI(object):
    """
    VQECI
        Args:
            mol:
                SCF or Mole to define the problem size.
            fermion_qubit_mapping (quri_parts.openfermion.transforms.OpenFermionQubitMapping):
                Mapping from :class:`FermionOperator` or :class:`InteractionOperator` to :class:`Operator`
            optimizer
            backend (Backend)
            shots_per_iter (int)
            initial_states (optional): Initial states for VQE
            ansatz: ansatz used for VQE
            layers (int):
                Layers of gate operations. Used for ``HardwareEfficient``, ``SymmetryPreserving``, ``ParticleConservingU1``, ``ParticleConservingU2``, and ``GateFabric``.
            k (int):
                Number of repetitions of excitation gates. Used for ``KUpCCGSD``.
            trotter_number (int):
                Number of trotter decomposition. Used for ``UCCSD`` and ``kUpCCGSD``.
            include_pi (bool):
                If ``True``, the optional constant gate is inserted. Used for ``GateFabric``.
            use_singles: (bool):
                If ``True``, single-excitation gates are applied. Used for ``UCCSD``.
            delta_sz (int):
                Changes of spin in the excitation. Used for ``KUpCCGSD``.
            singlet_excitation (bool):
                If ``True``, the ansatz will be spin symmetric. Used for ``UCCSD`` and
                ``KUpCCGSD``.
            is_init_random (bool):
                If ``False``, initial parameters are initialized to 0s, else, initialized randomly.
            seeed (int):
                Random seed.
        Return:
            None
    """

    def __init__(
        self,
        mol=None,
        fermion_qubit_mapping=jordan_wigner,
        optimizer=Adam(),
        backend: Backend = QulacsBackend(),
        shots_per_iter: int = 10000,
        initial_states=None,
        ansatz: Ansatz = Ansatz.UCCSD,
        layers: int = 2,
        k: int = 1,
        trotter_number: int = 1,
        excitation_number: int = 0,
        weight_policy: str = "exponential",
        include_pi: bool = False,
        use_singles: bool = True,
        delta_sz: int = 0,
        singlet_excitation: bool = False,
        is_init_random: bool = False,
        seed: int = 0,
    ):
        self.mol = mol
        self.fermion_qubit_mapping = fermion_qubit_mapping
        self.opt_param = None  # to be used to store the optimal parameter for the VQE
        self.opt_states: list = [None]
        self.n_qubit: int = None
        self.n_orbitals: int = None
        self.initial_states = initial_states
        self.ansatz: Ansatz = ansatz
        self.optimizer = optimizer
        self.n_electron: int = None
        self.layers: int = layers
        self.k: int = k
        self.trotter_number: int = trotter_number
        self.include_pi: bool = include_pi
        self.use_singles: bool = use_singles
        self.delta_sz: int = delta_sz
        self.singlet_excitation: bool = singlet_excitation
        self.is_init_random: bool = is_init_random
        self.seed: int = seed
        self.e = 0
        self.excitation_number = excitation_number
        self.weight_policy = weight_policy

        self.energies: list = None

        self.energies: list = None
        self.max_iter = 100000
        self.sim = None  # Initialized later
        self.opt_states_vec:list=None
        self.sim=None
        self.total_circuit = None
    # =======================================================================================
    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        self.n_orbitals = norb
        self.n_qubit = self.fermion_qubit_mapping.n_qubits_required(2 * self.n_orbitals)
        self.n_electron = nelec[0] + nelec[1]
        self.sim=Simulator('mqvector', self.n_qubit)
        # Get the active space Hamiltonian
        active_hamiltonian = _get_active_hamiltonian(h1, h2, norb, ecore)
        # Convert the Hamiltonian using `self.fermion_qubit_mapping`
        self.fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        self.fermionic_hamiltonian_ms=FermionOperator_ms.from_openfermion(self.fermionic_hamiltonian)
        op_mapper = self.fermion_qubit_mapping.get_of_operator_mapper(
            n_spin_orbitals=2 * self.n_orbitals,
            n_fermions=self.n_electron,
        )
        from openfermion.transforms import jordan_wigner
        qubit_hamiltonian=jordan_wigner(self.fermionic_hamiltonian)
        # qubit_hamiltonian = op_mapper(
        #     self.fermionic_hamiltonian,
        # )

        qubit_hamiltonian=QubitOperator_ms.from_openfermion(qubit_hamiltonian)
        # Set initial Quantum State

        if self.initial_states is None:
            self.initial_states = generate_initial_states(
                self.n_orbitals,
                self.n_electron,
                self.excitation_number,
                self.fermion_qubit_mapping,
            )
        if not isinstance(self.initial_states, list):
            raise TypeError("Initial_states must be of type list.")

        # Set given ansatz
        ansatz = _create_ansatz(
            self.ansatz,
            self.fermion_qubit_mapping,
            2 * self.n_orbitals,
            self.n_electron,
            self.layers,
            self.k,
            self.trotter_number,
            self.include_pi,
            self.use_singles,
            self.delta_sz,
            self.singlet_excitation,
        )
        ansatz = quri_to_mindquantum(ansatz)
        # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        # print(ansatz)

        init_amplitudes=np.zeros(len(ansatz.params_name))

        # Hartree-Fock initial circuit
        from mindquantum.core.gates import X
        hf_circuit = Circuit_ms([X.on(i) for i in range(self.n_electron)])
        self.total_circuit = hf_circuit+ansatz




        # VQE network
        # sim_temp = Simulator('mqvector', self.n_qubit)  # Temp sim for grad
        molecule_pqc = self.sim.get_expectation_with_grad(Hamiltonian_ms(qubit_hamiltonian), self.total_circuit)

        # Optimization function (matching example)
        def fun(p, molecule_pqc, energy_list=None):
            f, g = molecule_pqc(p)
            f = np.real(f)[0, 0]
            g = np.real(g)[0, 0]
            if energy_list is not None:
                energy_list.append(f)
                if len(energy_list) % 5 == 0:
                    print(f"Step: {len(energy_list)},\tenergy: {f}")
            return f, g

        # Initial params: use init_amplitudes from generate_uccsd
        init_params = init_amplitudes
        if self.is_init_random:
            np.random.seed(self.seed)
            init_params = np.random.random(size=len(init_params))

        # Optimize with SciPy (BFGS, as in example)
        print("----VQE-----")
        energy_list = []
        res = minimize(fun, init_params, args=(molecule_pqc, energy_list), method='bfgs', jac=True, options={'maxiter': self.max_iter})

        self.opt_param = res.x

        

        # Get energy
        self.e = res.fun
        self.energies = [self.e]

        # Optimal state
        self.sim.reset()
        self.sim.apply_circuit(self.total_circuit, self.opt_param)
        self.opt_states = [self.total_circuit.apply_value(self.opt_param)]
        self.opt_states_vec=[self.sim.get_qs()]

        return self.energies[0], None


    # ======================
    def make_rdm1(self, _, norb, nelec, link_index=None, sz=0, **kwargs):

        nelec = sum(nelec)
        dm1 = self._one_rdm(self.opt_states[0], norb, nelec, sz)
        return dm1

    # ======================
    def make_rdm12(self, _, norb, nelec, link_index=None, sz=0, **kwargs):
        nelec = sum(nelec)
        dm2 = self._two_rdm(self.opt_states[0], norb, nelec)
        return self._one_rdm(self.opt_states[0], norb, nelec, sz), dm2

    # ======================
    def spin_square(self, civec, norb, nelec):
        return 0, 1

    # ======================
    def _one_rdm(self, state, norb, nelec, sz=0):
        vqe_one_rdm = np.zeros((norb, norb))
        # get 1 rdm
        spin_dependent_rdm = np.real(
            get_1rdm(self.total_circuit, self.opt_param, self.n_qubit, nelec, sz)
        )
        # transform it to spatial rdm
        vqe_one_rdm += spin_dependent_rdm[::2, ::2] + spin_dependent_rdm[1::2, 1::2]
        self.my_one_rdm = vqe_one_rdm
        # print("vqe_one_rdm_done")
        # import sys
        # sys.exit()
        return vqe_one_rdm

    # ======================
    def _dm2_elem(self, i, j, k, m, state, norb, nelec):
        op = FermionOperator(((i, 1), (j, 1), (k, 0), (m, 0)))
        qubit_op = jordan_wigner(op)
        ms_qubit_op = QubitOperator_ms.from_openfermion(qubit_op)
        ham = Hamiltonian_ms(ms_qubit_op)
        
        sim = Simulator('mqvector', self.n_qubit)
        sim.apply_circuit(self.total_circuit, pr=self.opt_param)
        two_rdm_real = sim.get_expectation(ham).real
        
        #
        # pyscf use real spin-free RDM (i.e. RDM in spatial orbitals)
        #
        return two_rdm_real

    # ======================
    def _two_rdm(self, state, norb, nelec):
        vqe_two_rdm = np.zeros((norb, norb, norb, norb))
        dm2aa = np.zeros_like(vqe_two_rdm)
        dm2ab = np.zeros_like(vqe_two_rdm)
        dm2bb = np.zeros_like(vqe_two_rdm)

        # generate 2 rdm
        spin_dependent_rdm = np.real(
            get_2rdm(
                self.total_circuit,
                self.opt_param,
                self.n_qubit,
                nelec,
            )
        )
        # import sys
        # sys.exit()


        # convert it into spatial
        for i, j, k, l in product(range(norb), range(norb), range(norb), range(norb)):
            ia = 2 * i
            ja = 2 * j
            ka = 2 * k
            la = 2 * l
            ib = 2 * i + 1
            jb = 2 * j + 1
            kb = 2 * k + 1
            lb = 2 * l + 1
            # aa
            dm2aa[i, j, k, l] = spin_dependent_rdm[ia, ja, ka, la]
            # bb
            dm2bb[i, j, k, l] = spin_dependent_rdm[ib, jb, kb, lb]
            #
            dm2ab[i, j, k, l] = spin_dependent_rdm[ia, jb, kb, la]
        self.my_two_rdm = (
            dm2aa.transpose(0, 3, 1, 2)
            + dm2bb.transpose(0, 3, 1, 2)
            + dm2ab.transpose(0, 3, 1, 2)
            + (dm2ab.transpose(0, 3, 1, 2)).transpose(2, 3, 0, 1)
        )
        return self.my_two_rdm

    # ======================
    def make_dm2(self, _, norb, nelec, link_index=None, **kwargs):
        dm2 = np.zeros((norb, norb, norb, norb))
        for i, j, k, l in product(range(norb), range(norb), range(norb), range(norb)):
            ia = 2 * i
            ja = 2 * j
            ka = 2 * k
            la = 2 * l
            ib = 2 * i + 1
            jb = 2 * j + 1
            kb = 2 * k + 1
            lb = 2 * l + 1

            dm2[i, j, k, l] = (
                self._dm2_elem(ia, ja, ka, la, self.opt_states[0], norb, nelec)
                + self._dm2_elem(ib, jb, kb, lb, self.opt_states[0], norb, nelec)
                + self._dm2_elem(ia, ja, kb, lb, self.opt_states[0], norb, nelec)
                + self._dm2_elem(ib, jb, ka, la, self.opt_states[0], norb, nelec)
            )
        return dm2