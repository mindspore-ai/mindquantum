# Copyright 2026 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Correctness tests for the circuit-tiled density matrix simulator."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core import gates as G  # noqa: N812
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator

# The density matrix of 7 qubits holds 4**7 amplitudes, above the OpenMP
# threshold, so it exercises the planned, parallel tile path.
_PARALLEL_QUBITS = 7
_SERIAL_QUBITS = 4


def _unitary(dim, seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    unitary, triangular = np.linalg.qr(raw)
    diagonal = np.diag(triangular)
    return unitary * (diagonal / np.abs(diagonal)).conj()


def _random_density_matrix(n_qubits, seed):
    rng = np.random.default_rng(seed)
    dim = 1 << n_qubits
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = raw @ raw.conj().T
    return rho / np.trace(rho)


def _kraus_matrix(m, qubits, n_qubits):
    """Embed a 2^k x 2^k operator acting on `qubits` (little endian) into n qubits."""
    dim = 1 << n_qubits
    out = np.zeros((dim, dim), dtype=np.complex128)
    k = len(qubits)
    for row in range(dim):
        row_bits = [(row >> q) & 1 for q in qubits]
        local_row = sum(bit << i for i, bit in enumerate(row_bits))
        rest = row
        for q in qubits:
            rest &= ~(1 << q)
        for local_col in range(1 << k):
            col = rest
            for i, q in enumerate(qubits):
                if (local_col >> i) & 1:
                    col |= 1 << q
            out[row, col] = m[local_row, local_col]
    return out


def _apply_kraus(rho, kraus_set, qubits, n_qubits):
    result = np.zeros_like(rho)
    for kraus in kraus_set:
        big = _kraus_matrix(np.asarray(kraus), qubits, n_qubits)
        result += big @ rho @ big.conj().T
    return result


def _evolve_reference(rho, gate, n_qubits, parameters):
    """Apply one gate or channel to a density matrix with dense numpy algebra."""
    if isinstance(gate, G.GroupedPauliChannel):
        for qubit, kraus_set in zip(gate.obj_qubits, gate.matrix()):
            rho = _apply_kraus(rho, kraus_set, [qubit], n_qubits)
        return rho
    if isinstance(gate, G.NoiseGate):
        return _apply_kraus(rho, gate.matrix(), gate.obj_qubits, n_qubits)
    touched = sorted(set(gate.obj_qubits + gate.ctrl_qubits))
    remap = {q: i for i, q in enumerate(touched)}
    # Circuit.matrix() is defined on the circuit's own qubits [0, n_touched).
    mapped = Circuit([gate.on([remap[q] for q in gate.obj_qubits], [remap[q] for q in gate.ctrl_qubits])])
    local = mapped.matrix(parameters) if mapped.params_name else mapped.matrix()
    full = _kraus_matrix(local, touched, n_qubits)
    return full @ rho @ full.conj().T


def _known_gate_circuit():  # pylint: disable=too-many-statements
    """Every gate and channel the tile builder lowers, with external controls and diagonal operands."""
    circuit = Circuit()
    circuit += G.H.on(0, 6)
    circuit += G.X.on(1, 5)
    circuit += G.Y.on(2)
    circuit += G.Z.on(3, 6)
    circuit += G.S.on(4)
    circuit += G.S.hermitian().on(5, 0)
    circuit += G.T.on(6)
    circuit += G.T.hermitian().on(0, 4)
    circuit += G.SX.on(1)
    circuit += G.SX.hermitian().on(2, 6)
    circuit += G.RX('rx').on(3, 5)
    circuit += G.RY('ry').on(4, 6)
    circuit += G.RZ('rz').on(5, 1)
    circuit += G.Rxx('rxx').on([0, 2], 6)
    circuit += G.Ryy('ryy').on([1, 3], 5)
    circuit += G.Rzz('rzz').on([2, 4], 6)
    circuit += G.Rxy('rxy').on([3, 5], 0)
    circuit += G.Rxz('rxz').on([4, 6], 1)
    circuit += G.Ryz('ryz').on([5, 0], 2)
    circuit += G.SWAP.on([0, 6], 3)
    circuit += G.ISWAP.on([1, 5], 6)
    circuit += G.ISWAP.hermitian().on([2, 4], 0)
    circuit += G.SWAPalpha('swap_alpha').on([3, 6], 1)
    circuit += G.Givens('givens').on([4, 5], 2)
    circuit += G.GroupedPauli('XYZIZ').on([0, 2, 4, 5, 6], 1)
    circuit += G.RotPauliString('XYZIZ', 'rps').on([1, 3, 5, 0, 6], 2)
    circuit += G.PhaseShift('phase').on(6, 4)
    circuit += G.GlobalPhase('global_phase').on(0, 5)
    circuit += G.U3('u3_theta', 'u3_phi', 'u3_lambda').on(2, 6)
    circuit += G.Rn('rn_alpha', 'rn_beta', 'rn_gamma').on(1, 3)
    circuit += G.FSim('fsim_theta', 'fsim_phi').on([0, 4], 6)
    circuit += G.UnivMathGate('custom_1q', _unitary(2, 101)).on(5, 2)
    circuit += G.UnivMathGate('custom_3q', _unitary(8, 102)).on([6, 1, 0], 4)
    circuit += G.DepolarizingChannel(0.05).on(0)
    circuit += G.DepolarizingChannel(0.1, 2).on([1, 4])
    circuit += G.AmplitudeDampingChannel(0.2).on(2)
    circuit += G.AmplitudeDampingChannel(0.15).hermitian().on(3)
    circuit += G.PhaseDampingChannel(0.3).on(4)
    circuit += G.PauliChannel(0.1, 0.2, 0.3).on(5)
    circuit += G.PauliChannel(0.0, 0.0, 0.3).on(0)
    circuit += G.BitFlipChannel(0.25).on(6)
    circuit += G.GroupedPauliChannel(np.array([[0.1, 0.05, 0.02], [0.03, 0.04, 0.2]])).on([0, 3])
    circuit += G.KrausChannel('kraus', G.DepolarizingChannel(0.4).matrix()).on(1)
    circuit += G.ThermalRelaxationChannel(100.0, 60.0, 5.0).on(2)
    circuit += G.ThermalRelaxationChannel(50.0, 80.0, 5.0).on(3)
    circuit += G.H.on(6)
    return circuit


def _parameters(circuit):
    names = circuit.params_name
    return dict(zip(names, np.linspace(-1.7, 2.1, len(names))))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_tiled_circuit_matches_dense_reference(dtype):
    """
    Feature: Circuit-tiled density matrix evolution.
    Description: Evolve every supported gate and channel through the parallel tile path on a mixed state.
    Expectation: The result matches a dense numpy evolution within the precision tolerance.
    """
    circuit = _known_gate_circuit()
    parameters = _parameters(circuit)
    rho = _random_density_matrix(_PARALLEL_QUBITS, 20260801)

    reference = rho.copy()
    for gate in circuit:
        reference = _evolve_reference(reference, gate, _PARALLEL_QUBITS, parameters)

    simulator = Simulator('mqmatrix', _PARALLEL_QUBITS, dtype=dtype)
    simulator.set_qs(rho)
    simulator.apply_circuit(circuit, parameters)

    atol = 2e-5 if dtype == mq.complex64 else 1e-11
    assert np.allclose(simulator.get_qs(), reference, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("n_qubits", [_SERIAL_QUBITS, _PARALLEL_QUBITS])
def test_gate_by_gate_matches_dense_reference(n_qubits):
    """
    Feature: Single-gate tiles.
    Description: Apply each gate and channel on its own, below and above the parallel threshold.
    Expectation: Every intermediate density matrix matches the dense reference.
    """
    circuit = Circuit([gate for gate in _known_gate_circuit() if max(gate.obj_qubits + gate.ctrl_qubits) < n_qubits])
    parameters = _parameters(circuit)
    rho = _random_density_matrix(n_qubits, 20260802)

    simulator = Simulator('mqmatrix', n_qubits)
    simulator.set_qs(rho)
    reference = rho.copy()
    for gate in circuit:
        reference = _evolve_reference(reference, gate, n_qubits, parameters)
        simulator.apply_gate(gate, parameters)
        assert np.allclose(simulator.get_qs(), reference, atol=1e-11), str(gate)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize(
    "n_qubits, gate",
    [
        (6, G.UnivMathGate('custom_5q', _unitary(32, 103)).on([5, 0, 3, 1, 4], 2)),
        (9, G.GroupedPauli('XYXZXYXYX').on(list(range(9)))),
        (9, G.RotPauliString('XYZXYZXY', 0.37).on(list(range(8)), 8)),
    ],
)
def test_oversized_gate_runs_in_its_own_tile(n_qubits, gate):
    """
    Feature: Gates mixing more index bits than a shared tile holds.
    Description: Apply gates whose doubled footprint exceeds the tile budget: a dense five-qubit unitary and
        nine-qubit Pauli strings.
    Expectation: The result matches the dense reference.
    """
    rho = _random_density_matrix(n_qubits, 20260803)
    circuit = Circuit([G.H.on(2), gate, G.H.on(2)])

    simulator = Simulator('mqmatrix', n_qubits)
    simulator.set_qs(rho)
    simulator.apply_circuit(circuit)
    reference = rho
    for step in circuit:
        reference = _evolve_reference(reference, step, n_qubits, {})
    assert np.allclose(simulator.get_qs(), reference, atol=1e-11)


def _gradient_circuit():
    """Build a circuit covering every built-in derivative supported by the tile engine."""
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += G.X.on(1, 6)
    circuit += G.RX({'shared': 1.7, 'rx': -0.3}).on(0, 5)
    circuit += G.RY('ry').on(1, 6)
    circuit += G.RZ('rz').on(2, 5)
    circuit += G.Rxx('rxx').on([0, 3], 6)
    circuit += G.Ryy('ryy').on([1, 4], 5)
    circuit += G.Rzz('rzz').on([2, 5], 6)
    circuit += G.Rxy('rxy').on([3, 6], 5)
    circuit += G.Rxz('rxz').on([4, 0], 6)
    circuit += G.Ryz('ryz').on([0, 5], 3)
    circuit += G.SWAPalpha('swap_alpha').on([1, 6], 2)
    circuit += G.Givens('givens').on([2, 4], 5)
    circuit += G.RotPauliString('XYZIZ', 'rps').on([0, 2, 4, 6, 5], 1)
    circuit += G.PhaseShift({'phase': 0.8, 'shared': -0.2}).on(5, 3)
    circuit += G.GlobalPhase('global_phase').on(0, 6)
    circuit += G.U3('u3_theta', 'u3_phi', 'u3_lambda').on(6, 4)
    circuit += G.Rn('rn_alpha', 'rn_beta', 'rn_gamma').on(3, 5)
    circuit += G.FSim('fsim_theta', 'fsim_phi').on([3, 5], 6)
    circuit += G.SX.on(4)
    circuit += G.H.on(6, 2)
    return circuit


def _hamiltonians(dtype):
    """Build representative Hamiltonians for adjoint-gradient comparisons."""
    first = Hamiltonian(
        0.71 * QubitOperator('Z0')
        - 0.23 * QubitOperator('X2 Y5')
        + 0.17 * QubitOperator('Z3 Z6')
        + 0.09 * QubitOperator('')
    ).astype(dtype)
    second = Hamiltonian(
        -0.41 * QubitOperator('Y1') + 0.37 * QubitOperator('X3 Z6') + 0.19 * QubitOperator('X4')
    ).astype(dtype)
    return [first, second]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_noiseless_gradient_matches_state_vector(dtype):
    """
    Feature: Adjoint differentiation on a density matrix.
    Description: Differentiate a pure-state circuit with every built-in derivative through the tile engine.
    Expectation: Values and gradients match the state vector simulator.
    """
    circuit = _gradient_circuit()
    parameters = np.linspace(-1.3, 1.7, len(circuit.params_name))
    hamiltonians = _hamiltonians(dtype)
    rng = np.random.default_rng(20260804)
    state = rng.normal(size=1 << _PARALLEL_QUBITS) + 1j * rng.normal(size=1 << _PARALLEL_QUBITS)
    state /= np.linalg.norm(state)

    reference = Simulator('mqvector', _PARALLEL_QUBITS, dtype=dtype)
    reference.set_qs(state)
    expected_value, expected_gradient = reference.get_expectation_with_grad(hamiltonians, circuit)(parameters)

    density = Simulator('mqmatrix', _PARALLEL_QUBITS, dtype=dtype)
    density.set_qs(state)
    value, gradient = density.get_expectation_with_grad(hamiltonians, circuit)(parameters)

    atol = 2e-4 if dtype == mq.complex64 else 1e-9
    assert np.allclose(value, expected_value, atol=atol)
    assert np.allclose(gradient, expected_gradient, atol=atol)


def _noisy_gradient_circuit():
    """Build a circuit interleaving parameterized gates and noise channels."""
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += G.RX({'a': 1.0, 'b': 2.0}).on(1, 0)
    circuit += G.DepolarizingChannel(0.05).on(1)
    circuit += G.RY('c').on(2)
    circuit += G.AmplitudeDampingChannel(0.2).on(0)
    circuit += G.Rzz('d').on([1, 2])
    circuit += G.KrausChannel('kraus', G.AmplitudeDampingChannel(0.3).matrix()).on(2)
    circuit += G.U3('e', 'f', 'g').on(0, 2)
    circuit += G.ThermalRelaxationChannel(100.0, 150.0, 20.0).on(1)
    circuit += G.RX('h').on(3, 1)
    circuit += G.PhaseDampingChannel(0.1).on(3)
    circuit += G.Rxx('a').on([0, 3])
    return circuit


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("n_qubits", [_SERIAL_QUBITS, _PARALLEL_QUBITS])
def test_noisy_gradient_matches_finite_differences(n_qubits):
    """
    Feature: Adjoint differentiation through non-unitary channels.
    Description: Differentiate a circuit interleaving parameterized gates with non-self-adjoint channels.
    Expectation: Values match a dense evolution and gradients match central finite differences.
    """
    circuit = _noisy_gradient_circuit()
    names = circuit.params_name
    parameters = np.linspace(-0.9, 1.1, len(names))
    hamiltonian = Hamiltonian(
        0.6 * QubitOperator('Z0')
        + 0.3 * QubitOperator('X1 Y2')
        - 0.2 * QubitOperator('Z3')
        + 0.5 * QubitOperator('Y0 X3')
    )
    rho = _random_density_matrix(n_qubits, 20260805)

    def expectation(values):
        pr = dict(zip(names, values))
        state = rho.copy()
        for gate in circuit:
            state = _evolve_reference(state, gate, n_qubits, pr)
        return np.trace(hamiltonian.hamiltonian.matrix(n_qubits) @ state).real

    simulator = Simulator('mqmatrix', n_qubits)
    simulator.set_qs(rho)
    value, gradient = simulator.get_expectation_with_grad(hamiltonian, circuit)(parameters)

    assert np.allclose(value, expectation(parameters), atol=1e-10)
    step = 1e-5
    for index, name in enumerate(names):
        shifted = parameters.copy()
        shifted[index] += step
        upper = expectation(shifted)
        shifted[index] -= 2 * step
        lower = expectation(shifted)
        assert np.isclose(gradient[0, 0, index], (upper - lower) / (2 * step), atol=1e-7), name


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_batched_hamiltonians_and_encoders():
    """
    Feature: Batched gradient evaluation.
    Description: Evaluate several Hamiltonians for several encoder inputs with a noisy circuit.
    Expectation: Every entry matches an independent single evaluation.
    """
    encoder = Circuit([G.RX('e0').on(0), G.RY('e1').on(1)]).as_encoder()
    ansatz = Circuit(
        [
            G.DepolarizingChannel(0.1).on(0),
            G.Rzz('a0').on([0, 1]),
            G.AmplitudeDampingChannel(0.2).on(1),
            G.RX('a1').on(1, 0),
        ]
    ).as_ansatz()
    circuit = encoder + ansatz
    hamiltonians = [
        Hamiltonian(QubitOperator('Z0')),
        Hamiltonian(QubitOperator('X0 Y1')),
        Hamiltonian(QubitOperator('Z1') + 0.5 * QubitOperator('X1')),
    ]
    encoder_data = np.array([[0.1, 0.2], [-0.7, 1.3], [2.1, -0.4]])
    ansatz_data = np.array([0.5, -1.1])

    simulator = Simulator('mqmatrix', 2)
    batched = simulator.get_expectation_with_grad(hamiltonians, circuit, parallel_worker=1)
    value, encoder_gradient, ansatz_gradient = batched(encoder_data, ansatz_data)

    for batch in range(encoder_data.shape[0]):
        for index, hamiltonian in enumerate(hamiltonians):
            single = simulator.get_expectation_with_grad(hamiltonian, circuit)
            expected_value, expected_encoder, expected_ansatz = single(encoder_data[batch], ansatz_data)
            assert np.allclose(value[batch, index], expected_value[0, 0], atol=1e-12)
            assert np.allclose(encoder_gradient[batch, index], expected_encoder[0, 0], atol=1e-12)
            assert np.allclose(ansatz_gradient[batch, index], expected_ansatz[0, 0], atol=1e-12)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_noisy_measurement_ending_sampling_uses_exact_distribution():
    """
    Feature: Sampling of measurement-ending noisy circuits.
    Description: Sample a noisy Bell circuit many times.
    Expectation: The empirical distribution matches the diagonal of the exact density matrix.
    """
    circuit = Circuit([G.H.on(0), G.X.on(1, 0), G.DepolarizingChannel(0.2).on(0), G.BitFlipChannel(0.1).on(1)])
    simulator = Simulator('mqmatrix', 2)
    simulator.apply_circuit(circuit)
    probabilities = np.real(np.diag(simulator.get_qs()))

    shots = 200000
    result = Simulator('mqmatrix', 2).sampling(circuit.measure_all(), shots=shots, seed=7)
    empirical = np.zeros(4)
    for key, count in result.data.items():
        empirical[int(key[::-1], 2)] = count / shots
    assert np.allclose(empirical, probabilities, atol=5e-3)


def _derivative_cases():
    """Build gate derivative cases and their dense reference matrices."""
    theta = 0.83
    d_ps = 1j * np.exp(1j * theta) * np.diag([0.0, 1.0])
    e_pi = np.exp(1j * np.pi * theta)
    d_swap_alpha = 0.5j * np.pi * e_pi * np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]])
    d_givens = np.array(
        [[0, 0, 0, 0], [0, -np.sin(theta), -np.cos(theta), 0], [0, np.cos(theta), -np.sin(theta), 0], [0, 0, 0, 0]]
    )
    pauli_string = np.kron(np.kron(np.kron(Z_MATRIX, Y_MATRIX), X_MATRIX), X_MATRIX)  # object order X X Y Z
    d_rps = -0.5 * np.sin(theta / 2) * np.eye(16) - 0.5j * np.cos(theta / 2) * pauli_string

    def controlled(matrix, n_controls):
        for _ in range(n_controls):
            matrix = np.kron(np.diag([0.0, 1.0]), matrix)
        return matrix

    ry = G.RY('theta').on(1, 2)
    return [
        # (gate, derivative operator on `qubits`, qubits)
        (ry, controlled(ry.diff_matrix({'theta': theta}), 1), [1, 2]),
        (G.RX('theta').on(1, [2, 0]), controlled(G.RX('theta').diff_matrix({'theta': theta}), 2), [1, 2, 0]),
        (G.PhaseShift('theta').on(0), d_ps, [0]),
        (G.PhaseShift('theta').on(0, 1), controlled(d_ps, 1), [0, 1]),
        (G.SWAPalpha('theta').on([0, 1]), d_swap_alpha, [0, 1]),
        (G.Givens('theta').on([2, 0]), d_givens, [2, 0]),
        # Five circuit qubits: the derivative tile exceeds the shared tile budget.
        (G.RotPauliString('XXYZ', 'theta').on([0, 1, 2, 3], 5), controlled(d_rps, 1), [0, 1, 2, 3, 5]),
    ]


X_MATRIX = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=complex)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
@pytest.mark.parametrize("case", _derivative_cases())
def test_apply_derivative_gate(dtype, case):
    """
    Feature: Derivative application on a density matrix.
    Description: Apply the derivative of a gate with `diff=True`, including derivatives that vanish on part of
        their operand space (unsatisfied controls, the |0> of a phase shift, the |00> and |11> of an exchange).
    Expectation: The state becomes dU rho dU^dagger, zero wherever dU vanishes.
    """
    gate, derivative, qubits = case
    n_qubits = 6
    rho = _random_density_matrix(n_qubits, 20260806)
    simulator = Simulator('mqmatrix', n_qubits, dtype=dtype)
    simulator.set_qs(rho)
    simulator.apply_gate(gate, {'theta': 0.83}, diff=True)

    full = _kraus_matrix(derivative, qubits, n_qubits)
    atol = 2e-5 if dtype == mq.complex64 else 1e-11
    assert np.allclose(simulator.get_qs(), full @ rho @ full.conj().T, atol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_multi_parameter_derivative_is_rejected():
    """
    Feature: Derivative application on a density matrix.
    Description: Apply a gate with several intrinsic derivatives with `diff=True`.
    Expectation: A ValueError, because the gate has no single derivative.
    """
    simulator = Simulator('mqmatrix', 1)
    with pytest.raises(ValueError, match="intrinsic derivatives"):
        simulator.apply_gate(G.U3('a', 'b', 'c').on(0), {'a': 0.1, 'b': 0.2, 'c': 0.3}, diff=True)


def _invalid_channels():
    from mindquantum import mqbackend as mb  # pylint: disable=import-outside-toplevel

    return [
        mb.gate.AmplitudeDampingChannel(False, 2.0, [0], []),
        mb.gate.AmplitudeDampingChannel(False, float('nan'), [0], []),
        mb.gate.AmplitudeDampingChannel(False, 0.1, [0, 1], []),
        mb.gate.PhaseDampingChannel(-0.5, [0], []),
        mb.gate.DepolarizingChannel(float('nan'), [0], []),
        mb.gate.DepolarizingChannel(2.0, [0], []),
        mb.gate.PauliChannel(0.8, 0.8, 0.8, [0], []),
        mb.gate.KrausChannel([[[2 + 0j, 0], [0, 2]]], [0], []),
        mb.gate.ThermalRelaxationChannel(100.0, 300.0, 10.0, [0], []),
        mb.gate.ThermalRelaxationChannel(-1.0, 1.0, 10.0, [0], []),
        mb.gate.AmplitudeDampingChannel(False, 0.1, [0], [1]),
    ]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_invalid_circuits_are_rejected_before_execution():
    """
    Feature: Compilation before execution.
    Description: Apply channels with unphysical parameters, alone and behind valid gates in a circuit.
    Expectation: Every invalid input raises a ValueError and leaves the state untouched.
    """
    expected = np.zeros((4, 4), dtype=complex)
    expected[0, 0] = 1
    for bad in _invalid_channels():
        simulator = Simulator('mqmatrix', 2)
        with pytest.raises(ValueError, match="requires"):
            simulator.backend.sim.apply_gate(bad)
        assert np.allclose(simulator.get_qs(), expected)

        simulator = Simulator('mqmatrix', 2)
        with pytest.raises(ValueError, match="requires"):
            simulator.backend.sim.apply_circuit([G.H.on(0).get_cpp_obj(), G.X.on(1, 0).get_cpp_obj(), bad])
        assert np.allclose(simulator.get_qs(), expected)

        simulator = Simulator('mqmatrix', 2)
        with pytest.raises(ValueError, match="requires"):
            simulator.backend.sim.sampling(
                [G.H.on(0).get_cpp_obj(), bad, G.Measure('m').on(0).get_cpp_obj()],
                mq.ParameterResolver(),
                10,
                {'m': 0},
                1,
            )
        assert np.allclose(simulator.get_qs(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_measurement_rejects_an_invalid_state():
    """
    Feature: Measurement.
    Description: Measure a density matrix whose trace is not finite.
    Expectation: A RuntimeError instead of a silently arbitrary outcome.
    """
    simulator = Simulator('mqmatrix', 1)
    simulator.backend.sim.set_dm(np.array([[complex(np.nan), 0], [0, 0]]))
    with pytest.raises(RuntimeError, match="trace"):
        simulator.apply_gate(G.Measure('m').on(0))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_gradient_worker_errors_propagate():
    """
    Feature: Batched gradient evaluation.
    Description: Evaluate a gradient on several worker threads with a parameter missing from the resolver.
    Expectation: The worker's exception is raised on the calling thread instead of terminating the process.
    """
    hamiltonian = [Hamiltonian(QubitOperator('Z0')).get_cpp_obj()]
    circuit = Circuit().rx('a', 0).as_encoder()
    simulator = Simulator('mqmatrix', 1)
    with pytest.raises(RuntimeError, match="parameter"):
        simulator.backend.sim.get_expectation_with_grad_multi_multi(
            hamiltonian, circuit.get_cpp_obj(), np.array([[0.1], [0.2], [0.3], [0.4]]), np.array([]), ['b'], [], 4, 1
        )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_mid_circuit_measurement_sampling():
    """
    Feature: Sampling with mid-circuit measurements.
    Description: Sample a circuit whose measurements are followed by further gates, including a reset.
    Expectation: The joint outcome statistics match the exact branch probabilities.
    """
    circuit = Circuit([G.H.on(0), G.X.on(1, 0), G.DepolarizingChannel(0.1).on(1)])
    circuit += G.Measure('first', reset_to=1).on(0)
    circuit += Circuit([G.H.on(1), G.Measure('second').on(0), G.Measure('third').on(1)])
    shots = 40000
    result = Simulator('mqmatrix', 2).sampling(circuit, shots=shots, seed=11)

    # Keys read `first second third`.  `first` is uniform, the reset pins
    # `second` to 1, and after the Hadamard `third` is uniform and independent.
    frequencies = {key: count / shots for key, count in result.data.items()}
    assert set(frequencies) == {'010', '011', '110', '111'}
    for frequency in frequencies.values():
        assert abs(frequency - 0.25) < 0.02


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_partial_trace_and_reduced_density_matrix_agree():
    """
    Feature: Partial trace.
    Description: Trace out qubits of a mixed state through both public entry points.
    Expectation: Both agree with a dense reshaping reference.
    """
    n_qubits = 5
    rho = _random_density_matrix(n_qubits, 20260807)
    simulator = Simulator('mqmatrix', n_qubits)
    simulator.set_qs(rho)
    kept = [0, 3]
    traced = [1, 2, 4]

    # Little endian: the row axes hold qubits n-1, ..., 0 followed by the column axes in the same order.
    reference = rho.reshape([2] * (2 * n_qubits))
    remaining = list(range(n_qubits - 1, -1, -1))
    for qubit in traced:
        row_axis = remaining.index(qubit)
        column_axis = len(remaining) + row_axis
        reference = np.trace(reference, axis1=row_axis, axis2=column_axis)
        remaining.remove(qubit)
    reference = reference.reshape(1 << len(kept), 1 << len(kept))

    assert np.allclose(simulator.get_partial_trace(traced), reference, atol=1e-12)
    assert np.allclose(simulator.get_reduced_density_matrix(kept), reference, atol=1e-12)
    assert np.allclose(simulator.get_reduced_density_matrix(kept[::-1]), reference, atol=1e-12)
