# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Device-resident Ascend state-vector simulation, without CPU simulation fallback."""

import copy

import numpy as np

from mindquantum.core import gates as G  # noqa: N812
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates.basic import NoiseGate, ParameterGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.operators.hamiltonian import HowTo
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.dtype import complex64, is_same_precision
from mindquantum.utils.string_utils import density_matrix_ket_string, ket_string
from mindquantum.utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_ansatz,
    _check_encoder,
    _check_gate_has_obj,
    _check_hamiltonian_qubits_number,
    _check_input_type,
    _check_int_type,
    _check_seed,
)

from .backend_base import BackendBase
from .utils import GradOpsWrapper

# These exact classes export native unitary semantics, not their Python matrix
# attributes. Subclasses may override those semantics and must still be checked.
_NATIVE_UNITARY_GATES = frozenset(
    (
        G.BarrierGate,
        G.HGate,
        G.IGate,
        G.XGate,
        G.YGate,
        G.ZGate,
        G.SGate,
        G.TGate,
        G.SXGate,
        G.CNOTGate,
        G.SWAPGate,
        G.ISWAPGate,
        G.GroupedPauli,
        G.RX,
        G.RY,
        G.RZ,
        G.Rxx,
        G.Ryy,
        G.Rzz,
        G.Rxy,
        G.Rxz,
        G.Ryz,
        G.Givens,
        G.SWAPalpha,
        G.GlobalPhase,
        G.PhaseShift,
        G.RotPauliString,
        G.U3,
        G.FSim,
        G.Rn,
    )
)


def _real_finite(values, name):
    """Reject complex or nonfinite parameters rather than discarding information."""
    values = np.asarray(values)
    if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain only real, finite values.')
    return values


def _check_resolver(pr):
    """Check both coefficients and the constant, including unused supplied values."""
    _real_finite([pr.const, *pr.values()], 'Parameters')


def _parameters(gate):
    """Get the intrinsic angle expressions, not just their shared parameter names."""
    return gate.get_parameters() if isinstance(gate, ParameterGate) else []


def _uncached_hamiltonian(ham):
    """Borrow current Python data for immediate export into native-owned snapshots."""
    result = copy.copy(ham)
    # Source data may have changed since either native cache was populated.
    # Clear only this lightweight wrapper's caches, never the caller's object.
    result.ham_cpp = result.herm_ham_cpp = None
    return result


def _shift_rule(gate, intrinsic, two_sided):
    """Exact Fourier shift rules, including the extra controlled-rotation frequency."""
    rotations = (G.RX, G.RY, G.RZ, G.Rxx, G.Ryy, G.Rzz, G.Rxy, G.Rxz, G.Ryz, G.RotPauliString)
    gate_type = type(gate)  # A subclass may implement a different Fourier spectrum.
    if gate_type in rotations or (gate_type is G.U3 and intrinsic == 0):
        frequencies = [0.5, 1.0] if two_sided and gate.ctrl_qubits else [1.0 if two_sided else 0.5]
    elif gate_type is G.Givens or (gate_type is G.FSim and intrinsic == 0):
        frequencies = [1.0, 2.0] if two_sided else [1.0]
    elif gate_type is G.SWAPalpha:
        frequencies = [np.pi]
    elif gate_type in (G.PhaseShift, G.GlobalPhase, G.U3, G.FSim):
        frequencies = [1.0]
    else:
        raise NotImplementedError(
            f'Exact parameter-shift is not implemented for {type(gate).__name__}. '
            'Use adjoint differentiation on a noiseless unitary circuit; no finite-difference fallback is used.'
        )
    frequencies = np.asarray(frequencies)
    shifts = (2 * np.arange(len(frequencies)) + 1) * np.pi / (2 * frequencies[-1])
    weights = np.linalg.solve(2 * np.sin(frequencies[:, None] * shifts), frequencies)
    return list(zip(shifts, weights))


def _gradient_parameters(inputs, encoders, ansatz):
    """Validate the complete batch before binding any circuit or launching device work."""
    both = bool(encoders and ansatz)
    if len(inputs) != (2 if both else 1):
        raise ValueError(f'Need {2 if both else 1} input arrays.')
    arrays = [_real_finite(value, 'Gradient inputs') for value in inputs]
    encoder = arrays[0] if encoders else np.empty((1, 0))
    if encoder.ndim == 1:
        encoder = encoder[None, :]
    weights = arrays[-1] if ansatz or not encoders else np.empty(0)
    _check_encoder(encoder, len(encoders))
    _check_ansatz(weights, len(ansatz))
    if encoder.shape[0] == 0:
        raise ValueError('Encoder batch cannot be empty.')
    return [ParameterResolver(dict(zip(encoders + ansatz, [*row, *weights]))) for row in encoder]


def _gradient_parameter_names(circuits):
    """Preserve each side's parameter order and reject contradictory roles."""
    encoders = tuple(dict.fromkeys(name for circuit in circuits for name in circuit.all_encoder.keys()))
    ansatz = tuple(dict.fromkeys(name for circuit in circuits for name in circuit.all_ansatz.keys()))
    if set(encoders) & set(ansatz):
        raise ValueError('A parameter cannot be both encoder and ansatz.')
    return encoders, ansatz


class MQVectorNPU(BackendBase):
    """
    Complex64 state-vector backend on Ascend, normally selected by ``Simulator('mqvector_npu', n_qubits)``.

    Importing this module does not load or initialize the NPU runtime. All state
    evolution, reductions, measurement, and sampling take place on the device.
    Host transfers are limited to explicitly requested states/matrices, scalar
    reductions, parameter/gate data, and measurement results.

    Noise uses normalized quantum trajectories. Noisy gradients use unbiased
    single-trajectory estimates of exact, gate-local parameter-shift rules, not
    finite differences. Average independent evaluations to reduce their variance.
    Batches are processed sequentially: each device operation already uses the
    available NPU cores. ``parallel_worker`` is validated but does not create CPU
    thread pools. Density-matrix-only operations are unsupported.

    Args:
        name (str): Backend name. A lone integer also supports ``MQVectorNPU(n_qubits)``.
        n_qubits (int): Number of logical qubits, subject to device memory.
        seed (int): Seed for independent measurement and trajectory randomness.
        dtype: Only ``mindquantum.complex64`` is supported; ``None`` selects it.
        device_id (int): Logical ACL device ordinal. Default: ``0``.
        tile_qubits (int): Maximum Vector tile width, between 4 and 12. Adjoint tiles are capped at 11. Default: ``12``.
        fusion_qubits (int): Planner support budget, at most ``tile_qubits - 3``. The adjoint and Cube engines cap
            their effective support at eight and six qubits respectively. Default: ``6``.
        cube_min_gates (int): Minimum gates per Cube region; zero disables Cube. Default: ``4``.
    """

    def __init__(
        self,
        name='mqvector_npu',
        n_qubits=None,
        seed=None,
        dtype=None,
        *,
        device_id=0,
        tile_qubits=12,
        fusion_qubits=6,
        cube_min_gates=4,
    ):
        """Validate all construction options before allocating a device state."""
        if not isinstance(name, str) and n_qubits is None:
            n_qubits, name = name, 'mqvector_npu'
        if name != 'mqvector_npu':
            raise ValueError('MQVectorNPU requires the backend name mqvector_npu.')
        dtype = complex64 if dtype is None else dtype
        if dtype != complex64:
            raise TypeError('mqvector_npu supports complex64 only; no implicit precision conversion is performed.')
        for key, value in (
            ('n_qubits', n_qubits),
            ('device_id', device_id),
            ('tile_qubits', tile_qubits),
            ('fusion_qubits', fusion_qubits),
            ('cube_min_gates', cube_min_gates),
        ):
            _check_int_type(key, value)
            if value < 0:
                raise ValueError(f'{key} must be nonnegative.')
        super().__init__(name, n_qubits, seed, dtype)
        # pylint: disable=import-outside-toplevel
        from mindquantum import _mq_vector_npu

        self._native = _mq_vector_npu
        self.device_id = device_id
        self.tile_qubits = tile_qubits
        self.fusion_qubits = fusion_qubits
        self.cube_min_gates = cube_min_gates
        self._native.validate_configuration(n_qubits, tile_qubits, fusion_qubits)
        self._rng = np.random.default_rng(self.seed)
        self.sim = self._native.State(n_qubits, device_id)

    @property
    def _policy(self):
        return self.tile_qubits, self.fusion_qubits, self.cube_min_gates

    @property
    def dtype(self):
        """Return the only supported arithmetic type."""
        return complex64

    def __repr__(self):
        """Describe the simulator without downloading its state."""
        return f'mqvector_npu({self.n_qubits} qubits, complex64, {self.sim.soc_name}, device_id={self.device_id})'

    __str__ = __repr__

    def _check_circuit(self, circuit):
        """Validate Python placement and stochastic operations before exporting deterministic regions."""
        _check_input_type('circuit', Circuit, circuit)
        if circuit.n_qubits > self.n_qubits:
            raise ValueError('Circuit has more qubits than the simulator.')
        keys = set()
        for gate in circuit:
            _check_gate_has_obj(gate)
            qubits = gate.obj_qubits + gate.ctrl_qubits
            for qubit in qubits:
                _check_int_type('qubit', qubit)
                if not 0 <= qubit < self.n_qubits:
                    raise ValueError(f'Gate {gate.name} has a qubit outside the simulator.')
            if len(set(qubits)) != len(qubits):
                raise ValueError('Object and control qubits must be distinct.')
            if isinstance(gate, (NoiseGate, G.Measure)) and gate.ctrl_qubits:
                raise ValueError('Noise and measurement gates cannot have control qubits.')
            if isinstance(gate, G.Measure):
                if gate.key in keys:
                    raise ValueError(f'Measure key {gate.key} already defined.')
                if len(gate.obj_qubits) != 1 or gate.reset_to not in (None, 0, 1):
                    raise ValueError('Invalid measurement qubit or reset value.')
                keys.add(gate.key)

    @staticmethod
    def _resolver(circuits, pr):
        """Resolve Python input types and names; native compilation validates gate expressions."""
        names = list(dict.fromkeys(name for circuit in circuits for name in circuit.params_name))
        if pr is None:
            if names:
                raise ValueError('A parameterized circuit requires complete parameter values.')
            pr = ParameterResolver()
        pr = _check_and_generate_pr_type(pr, names)
        _check_resolver(pr)
        # Gate expressions and generated coefficients are validated by the
        # native compiler before any state mutation. Do not recompute every
        # intrinsic angle through Python merely to bind a deterministic circuit.
        return pr

    @staticmethod
    def _cpp_gates(circuit):
        # Avoid Circuit's mutable C++ cache when a caller edits gate objects in place.
        return [gate.get_cpp_obj() for gate in circuit if not isinstance(gate, G.BarrierGate)]

    def _compile(self, gates, pr):
        return self.sim.prepare(self._native.compile(gates, pr, self.n_qubits, *self._policy))

    def compile(self, circuit, pr=None):
        """Snapshot a deterministic circuit into a reusable, device-resident executable."""
        self._check_circuit(circuit)
        pr = self._resolver([circuit], pr)
        if circuit.has_measure_gate or circuit.is_noise_circuit:
            raise ValueError('compile requires a circuit without measurement or noise.')
        return self._compile(self._cpp_gates(circuit), pr)

    def apply_compiled(self, executable, *, repetitions=1):
        """Apply a prepared executable on this state, without rebinding parameters."""
        _check_int_type('repetitions', repetitions)
        if repetitions < 1:
            raise ValueError('repetitions must be positive.')
        self.sim.apply(executable, repetitions)

    def _pauli_executables(self, qubits):
        return [
            [None, *(self._compile([pauli.on(qubit).get_cpp_obj()], ParameterResolver()) for pauli in (G.X, G.Y, G.Z))]
            for qubit in qubits
        ]

    def _noise_operation(self, gate):
        """Prepare all branches before any evolution, without joint Pauli Kraus expansion."""
        if isinstance(gate, (G.PauliChannel, G.GroupedPauliChannel)):
            probs = gate.probs if isinstance(gate, G.GroupedPauliChannel) else [[gate.px, gate.py, gate.pz]]
            probs = _real_finite(probs, 'Pauli probabilities')
            if probs.shape != (len(gate.obj_qubits), 3) or (probs < 0).any() or (probs.sum(axis=1) > 1).any():
                raise ValueError('Invalid Pauli probabilities.')
            probs = np.column_stack((1 - probs.sum(axis=1), probs))
            return 'pauli', (probs, self._pauli_executables(gate.obj_qubits))
        if isinstance(gate, G.DepolarizingChannel):
            probability = gate.p * (1 - 4.0 ** -len(gate.obj_qubits))
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Invalid depolarizing probability.')
            return 'depolarizing', (probability, self._pauli_executables(gate.obj_qubits))
        if not isinstance(
            gate, (G.AmplitudeDampingChannel, G.PhaseDampingChannel, G.KrausChannel, G.ThermalRelaxationChannel)
        ):
            raise NotImplementedError(f'Noise channel {type(gate).__name__} is not supported by mqvector_npu.')
        if isinstance(gate, G.ThermalRelaxationChannel):
            _real_finite([gate.t1, gate.t2, gate.gate_time], 'Thermal relaxation times')
            # Amplitude damping followed by dephasing gives the same thermal map.
            # Unlike the channel's eigendecomposition, this remains well-defined
            # when exp(-gate_time / t1) or exp(-gate_time / t2) underflows to zero.
            survival = np.exp(-gate.gate_time / gate.t1)
            dephasing_rate = (1 - 0.5 * (gate.t2 / gate.t1)) / gate.t2
            phase_probability = -0.5 * np.expm1(-gate.gate_time * dephasing_rate) if gate.gate_time else 0.0
            no_jump = np.diag([1.0, np.sqrt(survival)])
            matrices = np.asarray(
                [
                    np.sqrt(1 - phase_probability) * no_jump,
                    np.sqrt(phase_probability) * np.diag([1, -1]) @ no_jump,
                    np.array([[0, np.sqrt(1 - survival)], [0, 0]]),
                ]
            )
        else:
            matrices = np.asarray(gate.matrix())
        if matrices.ndim != 3 or matrices.shape[1:] != (2, 2) or not np.isfinite(matrices).all():
            raise ValueError('Kraus matrices must be finite single-qubit matrices.')
        executables = [
            self._compile([G.UnivMathGate('kraus', matrix).on(gate.obj_qubits).get_cpp_obj()], ParameterResolver())
            for matrix in matrices
        ]
        return 'kraus', executables

    def _operations(self, circuit, pr=None, *, planning=False):
        """Compile every deterministic region and noise branch before returning."""
        operations, region = [], []

        def flush():
            if region:
                cpp = self._cpp_gates(region)
                if planning:
                    operations.append(('forward', self._native.ForwardPlan(cpp, self.n_qubits, *self._policy)))
                else:
                    operations.append(('apply', self._compile(cpp, pr)))
                region.clear()

        for gate in circuit:
            if isinstance(gate, G.BarrierGate):
                continue
            if isinstance(gate, (G.Measure, NoiseGate)):
                flush()
                operations.append(('measure', gate) if isinstance(gate, G.Measure) else self._noise_operation(gate))
            else:
                region.append(gate)
        flush()
        return operations

    def _bind(self, operations, pr):
        return [
            ('apply', self.sim.prepare(data.bind(pr))) if kind == 'forward' else (kind, data)
            for kind, data in operations
        ]

    @staticmethod
    def _choice(weights, rng):
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0 or np.any(np.asarray(weights) < 0):
            raise ValueError('Cannot select a trajectory from a zero or nonfinite state.')
        return min(int(np.searchsorted(np.cumsum(weights), rng.random() * total, side='right')), len(weights) - 1)

    def _run(self, state, operations, rng):
        """Execute already prepared operations; only scalars/results leave the device."""
        measured = {}
        candidate = None
        for kind, data in operations:
            if kind == 'apply':
                state.apply(data)
            elif kind == 'measure':
                reset_to = -1 if data.reset_to is None else data.reset_to
                measured[data.key] = state.measure(data.obj_qubits[0], float(rng.random()), reset_to)
            elif kind == 'pauli':
                probs, executables = data
                for weights, choices in zip(probs, executables):
                    chosen = choices[self._choice(weights, rng)]
                    if chosen is not None:
                        state.apply(chosen)
            elif kind == 'depolarizing':
                probability, executables = data
                if rng.random() < probability:
                    # Uniformly choose a nonidentity Pauli string, including the p > 1 case.
                    code = int(rng.integers(1, 1 << (2 * len(executables)), dtype=np.uint64))
                    for choices in executables:
                        chosen = choices[code & 3]
                        code >>= 2
                        if chosen is not None:
                            state.apply(chosen)
            elif kind == 'kraus':
                masses = []
                for executable in data:
                    if candidate is None:
                        candidate = state.copy()
                    else:
                        candidate.copy_from(state)
                    candidate.apply(executable)
                    masses.append(candidate.norm())
                selected = self._choice(masses, rng)
                state.apply(data[selected])
                state.scale(1 / np.sqrt(masses[selected]))
        return measured

    @staticmethod
    def _measure_result(circuit):
        result = G.MeasureResult()
        result.add_measure([gate for gate in circuit if isinstance(gate, G.Measure)])
        return result

    def apply_circuit(self, circuit, pr=None):
        self._check_circuit(circuit)
        pr = self._resolver([circuit], pr)
        # The common unitary path is exactly one compile/prepare/apply, not a gate loop.
        if not circuit.has_measure_gate and not circuit.is_noise_circuit:
            self.sim.apply(self._compile(self._cpp_gates(circuit), pr))
            return None
        result = self._measure_result(circuit)
        operations = self._operations(circuit, pr)
        measured = self._run(self.sim, operations, self._rng)
        if measured:
            result.collect_data(np.array([[measured[key] for key in result.keys]], dtype=np.int8))
            return result
        return None

    def apply_gate(self, gate, pr=None, diff=False):
        _check_input_type('gate', G.BasicGate, gate)
        _check_input_type('diff', bool, diff)
        circuit = Circuit([gate])
        self._check_circuit(circuit)
        pr = self._resolver([circuit], pr)
        if isinstance(gate, G.BarrierGate):
            return None
        if diff:
            program = self._native.compile_derivative(gate.get_cpp_obj(), pr, self.n_qubits, self.tile_qubits)
            self.sim.apply(self.sim.prepare(program))
            return None
        result = self.apply_circuit(circuit, pr)
        return int(result.samples[0][0]) if isinstance(gate, G.Measure) else None

    def reset(self):
        self.sim.reset()

    def copy(self):
        result = copy.copy(self)
        result.sim = self.sim.copy()
        result._rng = copy.deepcopy(self._rng)
        return result

    def astype(self, dtype, seed=None):
        if dtype != complex64:
            raise TypeError('mqvector_npu supports complex64 only.')
        if seed is not None:
            _check_seed(seed)
        result = self.copy()
        if seed is not None:
            result.seed = seed
            result._rng = np.random.default_rng(seed)
        return result

    def get_qs(self, ket=False):
        _check_input_type('ket', bool, ket)
        state = self.sim.get_qs()
        return '\n'.join(ket_string(state)) if ket else state

    def entropy(self):
        return 0.0

    def purity(self):
        return 1.0

    def get_pure_state_vector(self):
        return self.get_qs()

    def _kept_qubits(self, qubits):
        qubits = [qubits] if isinstance(qubits, int) else qubits
        _check_input_type('qubits', (list, tuple), qubits)
        for qubit in qubits:
            _check_int_type('qubit', qubit)
            if not 0 <= qubit < self.n_qubits:
                raise ValueError('Qubit index is outside the simulator.')
        if len(set(qubits)) != len(qubits):
            raise ValueError('Kept qubits must be distinct.')
        return sorted(qubits)

    def get_reduced_density_matrix(self, kept_qubits):
        kept_qubits = self._kept_qubits(kept_qubits)
        if len(kept_qubits) >= self.n_qubits:
            raise ValueError('Number of kept qubits must be less than total number of qubits.')
        return np.asarray(self.sim.reduced_density(kept_qubits))

    def get_partial_trace(self, traced_qubits):
        """Partial-trace simulator objects require a density-matrix backend."""
        raise NotImplementedError('Use get_reduced_density_matrix to obtain a reduced matrix from mqvector_npu.')

    def get_qs_of_qubits(self, qubits, ket=False):
        _check_input_type('ket', bool, ket)
        qubits = self._kept_qubits(qubits)
        if len(qubits) == self.n_qubits:
            return self.get_qs(ket)
        density = self.get_reduced_density_matrix(qubits)
        if ket:
            return density_matrix_ket_string(density)
        if abs(np.trace(density @ density).real - 1) < 1e-6:
            # Only the requested reduced matrix is diagonalized on the host.
            _, vectors = np.linalg.eigh(density)
            vector = vectors[:, -1]
            if vector[np.flatnonzero(vector)[0]].real < 0:
                vector = -vector
            return vector
        return density

    def set_qs(self, quantum_state):
        state = np.asarray(quantum_state, dtype=np.complex128)
        if state.shape != (1 << self.n_qubits,):
            raise ValueError(f'Expected a state vector with shape {(1 << self.n_qubits,)}; got {state.shape}.')
        if not np.isfinite(state).all():
            raise ValueError('Quantum state amplitudes must be finite.')
        # Normalize host input robustly, including very large or very small amplitudes.
        scale = max(np.max(np.abs(state.real)), np.max(np.abs(state.imag)))
        if scale == 0:
            raise ValueError('Quantum state must have nonzero norm.')
        state = state / scale
        state /= np.linalg.norm(state)
        self.sim.set_qs(np.ascontiguousarray(state, dtype=np.complex64))

    def _compatible(self, other):
        _check_input_type('simulator_left', MQVectorNPU, other)
        if (self.n_qubits, self.device_id) != (other.n_qubits, other.device_id):
            raise ValueError('Both NPU states must have the same qubit count and device_id.')
        return other

    def inner_product(self, ket):
        """Return this bra's inner product with a compatible ket, reduced on device."""
        return self.sim.dot(self._compatible(ket).sim)

    def _check_hamiltonian(self, ham):
        _check_input_type('hamiltonian', Hamiltonian, ham)
        _check_hamiltonian_qubits_number(ham, self.n_qubits)
        if not is_same_precision(self.dtype, ham.dtype):
            raise TypeError('mqvector_npu requires a single-precision Hamiltonian; convert it explicitly first.')
        values = ham.sparse_matrix.data if ham.how_to == HowTo.FRONTEND else [coeff for _, coeff in ham.ham_termlist]
        if not np.isfinite(values).all():
            raise ValueError('Hamiltonian coefficients must be finite.')

    def apply_hamiltonian(self, hamiltonian):
        self._check_hamiltonian(hamiltonian)
        self.sim.apply_hamiltonian(self.sim.prepare_observable(hamiltonian.get_cpp_obj()))

    def get_expectation(self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None):
        self._check_hamiltonian(hamiltonian)
        circ_right = Circuit() if circ_right is None else circ_right
        same = circ_left is None and simulator_left is None
        circ_left = circ_right if circ_left is None else circ_left
        left = self if simulator_left is None else self._compatible(getattr(simulator_left, 'backend', simulator_left))
        for circuit in (circ_right, circ_left):
            self._check_circuit(circuit)
        pr = self._resolver([circ_right, circ_left], pr)
        right_ops = self._operations(circ_right, pr)
        left_ops = right_ops if same else self._operations(circ_left, pr)
        observable = self.sim.prepare_observable(hamiltonian.get_cpp_obj())
        # A read-only contraction needs no second full state. Copy only when
        # an explicitly supplied circuit will evolve the temporary state.
        ket = self.sim.copy() if right_ops else self.sim
        self._run(ket, right_ops, self._rng)
        if same:
            return ket.expectation(observable)
        bra = left.sim.copy() if left_ops else left.sim
        self._run(bra, left_ops, self._rng)
        return ket.expectation(observable, bra)

    def sampling(self, circuit, pr=None, shots=1, seed=None):
        self._check_circuit(circuit)
        if not circuit.has_measure_gate:
            raise ValueError('circuit must have at least one measurement gate.')
        _check_int_type('shots', shots)
        if shots < 1:
            raise ValueError('shots must be positive.')
        if pr is not None and not isinstance(pr, (dict, ParameterResolver)):
            raise TypeError('sampling pr requires a dict or ParameterResolver.')
        pr = self._resolver([circuit], pr)
        if seed is not None:
            _check_seed(seed)
        result = self._measure_result(circuit)
        operations = self._operations(circuit, pr)
        if seed is None:
            seed = int(self._rng.integers(0, 2**23))
        rng = np.random.default_rng(seed)
        state = self.sim.copy()
        first_measure = next(i for i, (kind, _) in enumerate(operations) if kind == 'measure')
        terminal = not circuit.is_noise_circuit and all(kind == 'measure' for kind, _ in operations[first_measure:])
        samples = np.empty((shots, len(result.keys)), dtype=np.int8)
        if terminal:
            self._run(state, operations[:first_measure], rng)
            indices = np.asarray(state.sample(shots, seed), dtype=np.uint64)
            for _, gate in operations[first_measure:]:
                bit = np.uint64(1 << gate.obj_qubits[0])
                samples[:, result.keys_map[gate.key]] = (indices & bit) != 0
                if gate.reset_to is not None:
                    indices = indices | bit if gate.reset_to else indices & ~bit
        else:
            for shot in range(shots):
                state.copy_from(self.sim)
                measured = self._run(state, operations, rng)
                samples[shot] = [measured[key] for key in result.keys]
        result.collect_data(samples)
        return result

    def get_circuit_matrix(self, circuit, pr=None):
        executable = self.compile(circuit, pr)
        dimension = 1 << self.n_qubits
        # Backend convention is rows of evolved basis states; Circuit.matrix transposes it.
        matrix = np.empty((dimension, dimension), dtype=np.complex64)
        basis = np.zeros(dimension, dtype=np.complex64)
        state = self.sim.copy()
        for column in range(dimension):
            basis[column] = 1
            state.set_qs(basis)
            state.apply(executable)
            matrix[column] = state.get_qs()
            basis[column] = 0
        return matrix

    @staticmethod
    def _unitary_validator(circuits):
        """Validate immutable constants once; check parameter-dependent custom matrices at every binding."""

        def check(gate, pr):
            matrix = gate.matrix(pr) if isinstance(gate, ParameterGate) else gate.matrix()
            matrix = np.asarray(matrix)
            if not np.isfinite(matrix).all() or not np.allclose(
                matrix.conj().T @ matrix, np.eye(len(matrix)), atol=2e-6, rtol=2e-6
            ):
                raise ValueError(f'Adjoint differentiation requires unitary gates; {gate.name} is not unitary.')

        dynamic = []
        for circuit in circuits:
            for gate in circuit:
                if type(gate) in _NATIVE_UNITARY_GATES:
                    continue
                if gate.parameterized:
                    dynamic.append(gate)
                else:
                    check(gate, {})

        def validate(pr):
            for gate in dynamic:
                check(gate, pr)

        return validate

    def _shift_variations(self, right, left, same, names):
        """Snapshot gate-local shifts and their Jacobians once, never shift shared input parameters."""
        variations = []
        sides = [('both', right)] if same else [('right', right), ('left', left)]
        for side, circuit in sides:
            for index, gate in enumerate(circuit):
                for intrinsic, coeff in enumerate(_parameters(gate)):
                    jacobian = [(names.index(name), float(coeff[name])) for name in coeff.requires_grad_parameters]
                    if not jacobian:
                        continue
                    for shift, weight in _shift_rule(gate, intrinsic, same):
                        for sign in (1, -1):
                            shifted = copy.deepcopy(circuit)
                            # Change only this intrinsic angle's constant. Parameter metadata is retained.
                            _parameters(shifted[index])[intrinsic].const += sign * shift
                            # Keep callback owners, not another entire circuit,
                            # alongside the native plans that use them.
                            keepalive = tuple(gate for gate in shifted if type(gate) not in _NATIVE_UNITARY_GATES)
                            variations.append(
                                (side, self._operations(shifted, planning=True), sign * weight, jacobian, keepalive)
                            )
        return variations

    def _shift_evaluator(self, hams, right, left, left_backend, same, names):
        """Prepare exact gate-local shifts and reuse call-local state buffers."""
        right_plan = self._operations(right, planning=True)
        left_plan = right_plan if left is right else self._operations(left, planning=True)
        variations = self._shift_variations(right, left, same, names)
        observables = [self.sim.prepare_observable(ham.get_cpp_obj()) for ham in hams]

        def evolve(source, plan, pr, state):
            operations = self._bind(plan, pr)
            if state is None:
                state = source.copy()
            else:
                state.copy_from(source)
            self._run(state, operations, self._rng)
            return state

        def evaluate(parameters):
            result = np.zeros((len(parameters), len(observables), 1 + len(names)), dtype=np.complex128)
            # Call-local ownership keeps independent invocations isolated.
            ket = bra = shifted = None
            for batch, pr in enumerate(parameters):
                ket = evolve(self.sim, right_plan, pr, ket)
                bra = ket if same else evolve(left_backend.sim, left_plan, pr, bra)
                result[batch, :, 0] = [ket.expectation(obs, bra) for obs in observables]
                for side, plan, weight, jacobian, _ in variations:
                    source = left_backend.sim if side == 'left' else self.sim
                    shifted = evolve(source, plan, pr, shifted)
                    shifted_ket = ket if side == 'left' else shifted
                    shifted_bra = bra if side == 'right' else shifted
                    values = np.asarray([shifted_ket.expectation(obs, shifted_bra) for obs in observables])
                    for parameter, coefficient in jacobian:
                        result[batch, :, 1 + parameter] += weight * coefficient * values
            return result

        evaluate.keepalive = (right, left)
        return evaluate

    def _adjoint_evaluator(self, hams, right, left, left_backend, same, names):
        """Own native plans and private Python callbacks independently of public metadata."""
        # Complex CSR operators need two sweeps even when the circuits coincide.
        hermitian = same and all(ham.how_to != HowTo.FRONTEND for ham in hams)
        custom_gates = []

        def snapshot_custom(circuit):
            for gate in circuit:
                if type(gate) not in _NATIVE_UNITARY_GATES:
                    gate = copy.deepcopy(gate)
                    custom_gates.append(gate)
                yield gate

        right_cpp = self._cpp_gates(snapshot_custom(right))
        left_cpp = []
        if not hermitian:
            left_cpp = right_cpp if left is right else self._cpp_gates(snapshot_custom(left))
        validate_unitary = self._unitary_validator((custom_gates,))
        plan = self._native.make_gradient(
            self.sim, [ham.get_cpp_obj() for ham in hams], right_cpp, left_cpp, names, hermitian, *self._policy
        )

        def evaluate(parameters):
            for pr in parameters:
                validate_unitary(pr)
            return np.asarray(plan.evaluate(self.sim, left_backend.sim, parameters))

        evaluate.keepalive = tuple(custom_gates)
        return evaluate

    def get_expectation_with_grad(
        self, hams, circ_right, circ_left=None, simulator_left=None, parallel_worker=None, pr_shift=False
    ):
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        _check_input_type('hams', list, hams)
        if not hams:
            raise ValueError('hams cannot be empty.')
        for ham in hams:
            self._check_hamiltonian(ham)
        _check_input_type('pr_shift', bool, pr_shift)
        if parallel_worker is not None:
            _check_int_type('parallel_worker', parallel_worker)
            if parallel_worker < 1:
                raise ValueError('parallel_worker must be positive.')
        same = circ_left is None and simulator_left is None
        left_backend = self if simulator_left is None else self._compatible(simulator_left)
        circ_left = circ_right if circ_left is None else circ_left
        circuits = (circ_right,) if circ_left is circ_right else (circ_right, circ_left)
        for circuit in circuits:
            self._check_circuit(circuit)
            if circuit.has_measure_gate:
                raise ValueError('Gradient circuits cannot contain measurement gates.')
        noisy = circ_right.is_noise_circuit or circ_left.is_noise_circuit
        if noisy and not same:
            raise ValueError('Noisy gradients do not support circ_left or simulator_left.')
        pr_shift = pr_shift or noisy
        encoders, ansatz = _gradient_parameter_names(circuits)
        names = encoders + ansatz
        n_encoders, n_hams = len(encoders), len(hams)
        uncached_hams = [_uncached_hamiltonian(ham) for ham in hams]
        if pr_shift:
            # Shifts/trajectories still need private Python gate data. Do not
            # duplicate an identical side or expose these snapshots as metadata.
            right = copy.deepcopy(circ_right)
            left = right if circ_left is circ_right else copy.deepcopy(circ_left)
            evaluate = self._shift_evaluator(uncached_hams, right, left, left_backend, same, names)
        else:
            evaluate = self._adjoint_evaluator(uncached_hams, circ_right, circ_left, left_backend, same, names)

        def grad_ops(*inputs):
            parameters = _gradient_parameters(inputs, encoders, ansatz)
            result = np.asarray(evaluate(parameters)).reshape((len(parameters), n_hams, 1 + len(names)))
            if encoders and ansatz:
                return result[:, :, 0], result[:, :, 1 : 1 + n_encoders], result[:, :, 1 + n_encoders :]
            return result[:, :, 0], result[:, :, 1:]

        # Public metadata is informational, as on the other backends. Execution
        # uses only native snapshots, private callback owners and frozen names.
        wrapper = GradOpsWrapper(
            grad_ops,
            hams,
            circ_right,
            circ_left,
            list(encoders),
            list(ansatz),
            parallel_worker,
            self,
        )
        wrapper.set_str(f'{self.n_qubits} qubits mqvector_npu VQA Operator')
        return wrapper
