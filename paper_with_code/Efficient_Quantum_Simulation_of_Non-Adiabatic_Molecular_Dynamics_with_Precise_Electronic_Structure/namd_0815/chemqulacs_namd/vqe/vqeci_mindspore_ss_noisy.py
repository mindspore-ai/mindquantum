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
from mindquantum import QubitNumberConstrain
import numpy as np
# from braket.aws import AwsDevice
from openfermion.ops import FermionOperator, InteractionOperator
from openfermion.transforms import get_fermion_operator, jordan_wigner

from pyscf import ao2mo
from quri_parts.algo.ansatz import HardwareEfficient, SymmetryPreserving
from quri_parts.algo.optimizer import Adam, OptimizerStatus,LBFGS
# from quri_parts.braket.backend import BraketSamplingBackend
from quri_parts.chem.ansatz import (
    AllSinglesDoubles,
    GateFabric,
    ParticleConservingU1,
    ParticleConservingU2,
)
from mindquantum.core.circuit import Circuit as Circuit_ms
from mindquantum.core.gates import X
from mindquantum.core.operators import FermionOperator as FermionOperator_ms
from mindquantum.core.operators import Hamiltonian as Hamiltonian_ms
from mindquantum.core.operators import QubitOperator as QubitOperator_ms
from mindquantum.simulator import Simulator
from mindquantum.algorithm.nisq import Transform as Transform_ms
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
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
from quri_parts.openfermion.ansatz import KUpCCGSD, TrotterUCCSD
from quri_parts.openfermion.transforms import (
    OpenFermionQubitMapping,
    jordan_wigner as jw_quri,
)
# from quri_parts.qiskit.backend import QiskitSamplingBackend
from quri_parts.qulacs.estimator import (
    create_qulacs_vector_concurrent_estimator,
    create_qulacs_vector_concurrent_parametric_estimator,
)

from .quri_to_mindquantum_circuit import quri_to_mindquantum
from mindquantum.core.circuit import DepolarizingChannelAdder, SequentialAdder,MixerAdder,QubitNumberConstrain,GateSelector
from mindquantum.simulator.noise import NoiseBackend


class Backend:
    """Base class represents backend"""

    pass


class QulacsBackend(Backend):
    """Backend class for Qulacs"""

    pass


class Ansatz(Enum):
    """An enum representing an ansatz for VQE"""

    HardwareEfficient = auto()
    SymmetryPreserving = auto()
    AllSinglesDoubles = auto()
    ParticleConservingU1 = auto()
    ParticleConservingU2 = auto()
    GateFabric = auto()
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


def generate_initial_states(
    n_orbitals, n_electron, excitation_number=0, fermion_qubit_mapping=jw_quri
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
        key=lambda lst: sum([2**a for a in lst])
    )[: excitation_number + 1]

    initial_states = []
    for occ_indices in occ_indices_lst:
        circ = Circuit_ms([X.on(i) for i in occ_indices])
        initial_states.append(circ)
    return initial_states, occ_indices_lst


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
    if ansatz == Ansatz.HardwareEfficient:
        return HardwareEfficient(n_qubits, layers)
    elif ansatz == Ansatz.SymmetryPreserving:
        return SymmetryPreserving(n_qubits, layers)
    elif ansatz == Ansatz.AllSinglesDoubles:
        return AllSinglesDoubles(n_qubits, n_electrons)
    elif ansatz == Ansatz.ParticleConservingU1:
        return ParticleConservingU1(n_qubits, layers)
    elif ansatz == Ansatz.ParticleConservingU2:
        return ParticleConservingU2(n_qubits, layers)
    elif ansatz == Ansatz.GateFabric:
        return GateFabric(n_qubits, layers, include_pi)
    elif ansatz == Ansatz.UCCSD:
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


adder = SequentialAdder([
    # 等价于 DepolarizingNoise(0.001, [], [])：应用于所有单比特门
    MixerAdder([
        QubitNumberConstrain(1),  # 限制为单比特门
        DepolarizingChannelAdder(p=0.001, n_qubits=1)  # 添加去极化噪声，概率 0.001
    ]),
    # 等价于 DepolarizingNoise(0.009, [], [gate_names.CNOT, gate_names.CZ])：应用于指定的双比特门（CNOT 和 CZ）
    MixerAdder([
        QubitNumberConstrain(2),  # 限制为双比特门
        GateSelector('cx'),  # 指定只针对 CNOT 和 CZ 门（MindQuantum 中门名为小写）
        DepolarizingChannelAdder(p=0.01, n_qubits=2)  # 添加去极化噪声，概率 0.009
    ]),    MixerAdder([
        QubitNumberConstrain(2),  # 限制为双比特门
        GateSelector('cz'),  # 指定只针对 CNOT 和 CZ 门（MindQuantum 中门名为小写）
        DepolarizingChannelAdder(p=0.01, n_qubits=2)  # 添加去极化噪声，概率 0.009
    ])
])


class VQECI:
    def __init__(
        self,
        mol,
        fermion_qubit_mapping: OpenFermionQubitMapping = jw_quri,
        initial_states=None,
        ansatz: Ansatz = Ansatz.UCCSD,
        optimizer=LBFGS(),
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
        backend=QulacsBackend(),
        shots_per_iter: int = 10000,
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

        self.estimator, self.parametric_estimator = create_qulacs_vector_concurrent_estimator(), create_qulacs_vector_concurrent_parametric_estimator()

        self.max_iter = 1000000
        self.sim = None  # Initialized later
        self.opt_states_vec:list=None
        self.total_circuits = None
        self.occ_indices_lst = None

    # =======================================================================================
    def kernel(self, h1, h2, norb, nelec, ecore=0, **kwargs):
        self.n_orbitals = norb
        self.n_qubit = self.fermion_qubit_mapping.n_qubits_required(2 * self.n_orbitals)
        self.n_electron = nelec[0] + nelec[1]

        # Add noise: 0.001 for single-qubit depolarizing, 0.01 for two-qubit depolarizing
        # single_dep = DepolarizingChannelAdder(0.001, n_qubits=1)
        # double_dep = DepolarizingChannelAdder(0.01, n_qubits=2)

        
        self.sim = Simulator('mqvector', self.n_qubit)

        # Get the active space Hamiltonian
        active_hamiltonian = _get_active_hamiltonian(h1, h2, norb, ecore)
        # Convert the Hamiltonian using `self.fermion_qubit_mapping`
        self.fermionic_hamiltonian = get_fermion_operator(active_hamiltonian)
        self.fermionic_hamiltonian_ms=FermionOperator_ms.from_openfermion(self.fermionic_hamiltonian)
        qubit_hamiltonian=jordan_wigner(self.fermionic_hamiltonian)
        # qubit_hamiltonian = op_mapper(
        #     self.fermionic_hamiltonian,
        # )

        qubit_hamiltonian=QubitOperator_ms.from_openfermion(qubit_hamiltonian)
        # Set initial Quantum State

        if self.initial_states is None:
            self.initial_states, self.occ_indices_lst = generate_initial_states(
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

        self.total_circuits = []
        for init_circ in self.initial_states:
            self.total_circuits.append(init_circ + ansatz)

        ham = Hamiltonian_ms(qubit_hamiltonian)
        molecule_pqcs = [self.sim.get_expectation_with_grad(ham, adder(circ)) for circ in self.total_circuits]
        # print(adder(self.total_circuits[0]))
        def cost_fn(params):
            fs = []
            for pqc in molecule_pqcs:
                f, _ = pqc(params)
                fs.append(np.real(f)[0, 0])
            if self.weight_policy == "exponential":
                weights = [2 ** (-i) for i in range(len(fs))]
            elif self.weight_policy == "same":
                weights = [1] * len(fs)
            elif self.weight_policy == "base_first":
                weights = [1] + [0.5] * (len(fs) - 1)
            else:
                raise ValueError(
                    "Invalid weight policy. weight_policy must be one of 'exponential', 'same', 'base_first'"
                )
            return sum(w * f for w, f in zip(weights, fs))

        def grad_fn(params):
            gs = []
            for pqc in molecule_pqcs:
                _, g = pqc(params)
                gs.append(np.real(g)[0, 0])
            if self.weight_policy == "exponential":
                weights = [2 ** (-i) for i in range(len(gs))]
            elif self.weight_policy == "same":
                weights = [1] * len(gs)
            elif self.weight_policy == "base_first":
                weights = [1] + [0.5] * (len(gs) - 1)
            else:
                raise ValueError(
                    "Invalid weight policy. weight_policy must be one of 'exponential', 'same', 'base_first'"
                )
            return np.sum([w * g for w, g in zip(weights, gs)], axis=0)

        print("----VQE-----")

        # init_params = np.zeros(len(ansatz.params_name))
        # if self.is_init_random:
        #     np.random.seed(self.seed)
        #     init_params = np.random.random(len(init_params))



        init_params = np.zeros(len(ansatz.params_name))
        if self.is_init_random:
            np.random.seed(self.seed)
            init_params = np.random.random(len(init_params))
        from scipy.optimize import minimize
        res = minimize(cost_fn, init_params, jac=grad_fn, method='L-BFGS-B', options={'maxiter': self.max_iter})
        print("Optimizer converged" if res.success else "Optimizer failed")

        self.opt_param = res.x

        # Store optimal state
        self.opt_states = [adder(circ).apply_value(self.opt_param) for circ in self.total_circuits]

        # Get energy
        self.e = res.fun
        self.energies = []
        for pqc in molecule_pqcs:
            f, _ = pqc(self.opt_param)
            self.energies.append(np.real(f)[0, 0])

        self.opt_states_vec = []
        for circ in self.total_circuits:
            self.sim.reset()
            self.sim.apply_circuit(adder(circ), self.opt_param)
            self.opt_states_vec.append(self.sim.get_qs(ket=False))  # Use ket=False for density matrix in noise simulation

        # result = vqe(init_params, cost_fn, grad_fn, self.optimizer)

        # self.opt_param = result.params

        # # Store optimal state
        # self.opt_states = [adder(circ).apply_value(self.opt_param) for circ in self.total_circuits]

        # # Get energy
        # self.e = result.cost
        # self.energies = []
        # for pqc in molecule_pqcs:
        #     f, _ = pqc(self.opt_param)
        #     self.energies.append(np.real(f)[0, 0])

        # self.opt_states_vec = []
        # for circ in self.total_circuits:
        #     self.sim.reset()
        #     self.sim.apply_circuit(adder(circ), self.opt_param)
        #     self.opt_states_vec.append(self.sim.get_qs(ket=False))  # Use ket=False for density matrix in noise simulation

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
        spin_dependent_rdm = np.zeros((self.n_qubit, self.n_qubit), dtype=complex)
        self.sim.reset()
        self.sim.apply_circuit(adder(self.total_circuits[0]), pr=self.opt_param)
        for p in range(self.n_qubit):
            for q in range(self.n_qubit):
                op = FermionOperator_ms(f"{p}^ {q}") + FermionOperator_ms(f"{q}^ {p}")
                qubit_op = Transform_ms(op).jordan_wigner()
                ham = Hamiltonian_ms(qubit_op)
                spin_dependent_rdm[p, q] = self.sim.get_expectation(ham)
        # transform it to spatial rdm
        vqe_one_rdm += np.real(spin_dependent_rdm[::2, ::2] + spin_dependent_rdm[1::2, 1::2]) / 2
        self.my_one_rdm = vqe_one_rdm
        return vqe_one_rdm

    # ======================
    def _dm2_elem(self, i, j, k, m, state, norb, nelec):
        op = FermionOperator_ms(((i, 1), (j, 1), (k, 0), (m, 0)))
        qubit_op = Transform_ms(op).jordan_wigner()
        ham = Hamiltonian_ms(qubit_op)
        
        # Removed reset and apply here to use the pre-applied state
        two_rdm_real = self.sim.get_expectation(ham).real
        
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

        self.sim.reset()
        self.sim.apply_circuit(adder(self.total_circuits[0]), pr=self.opt_param)

        # generate 2 rdm
        for i, j, k, l in product(range(norb), repeat=4):
            ia = 2 * i
            ja = 2 * j
            ka = 2 * k
            la = 2 * l
            ib = 2 * i + 1
            jb = 2 * j + 1
            kb = 2 * k + 1
            lb = 2 * l + 1
            # aa
            dm2aa[i, j, k, l] = self._dm2_elem(ia, ja, ka, la, state, norb, nelec)
            # bb
            dm2bb[i, j, k, l] = self._dm2_elem(ib, jb, kb, lb, state, norb, nelec)
            #
            dm2ab[i, j, k, l] = self._dm2_elem(ia, jb, kb, la, state, norb, nelec)
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
        self.sim.reset()
        self.sim.apply_circuit(adder(self.total_circuits[0]), pr=self.opt_param)
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
