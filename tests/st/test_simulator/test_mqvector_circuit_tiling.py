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
"""Correctness tests for mqvector circuit-tiling planning and execution."""

import platform
import sys

import numpy as np
import pytest

import mindquantum as mq
from mindquantum import _mq_vector
from mindquantum.core import gates as G  # noqa: N812
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator

_OPENMP_QUBIT_THRESHOLD = 13
_SEQUENTIAL_QUBITS = _OPENMP_QUBIT_THRESHOLD - 1


@pytest.mark.level0
def test_cpu_backend_selection_is_safe_for_the_host_platform():
    """
    Feature: Runtime CPU backend selection.
    Description: Inspect the backends selected for both state precisions on the current host platform.
    Expectation: Every selected backend is supported by the host architecture and operating system.
    """
    machine = platform.machine().lower()
    backends = dict(_mq_vector._cpu_backends)
    assert set(backends) == {'complex64', 'complex128'}
    if machine in ('aarch64', 'arm64'):
        assert set(backends.values()) <= {'baseline', 'arm-sve'}
        if sys.platform == 'darwin':
            assert set(backends.values()) == {'baseline'}
    else:
        assert set(backends.values()) == {'x86-avx2'}


def _normalized_state(n_qubits, seed):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << n_qubits) + 1j * rng.normal(size=1 << n_qubits)
    return state / np.linalg.norm(state)


def _unitary(dim, seed):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    unitary, triangular = np.linalg.qr(raw)
    diagonal = np.diag(triangular)
    return unitary * (diagonal / np.abs(diagonal)).conj()


def _idle_high_qubit(state):
    """Embed a state with an additional, untouched most-significant qubit in |0>."""
    return np.concatenate((state, np.zeros_like(state)))


def _gradient_hamiltonians(dtype):
    """Build representative Hamiltonians for adjoint-gradient comparisons."""
    first = Hamiltonian(
        0.71 * QubitOperator('Z0')
        - 0.23 * QubitOperator('X2 Y5')
        + 0.17 * QubitOperator('Z8 Z9')
        + 0.09 * QubitOperator('')
    ).astype(dtype)
    second = Hamiltonian(
        -0.41 * QubitOperator('Y1') + 0.37 * QubitOperator('X3 Z6') + 0.19 * QubitOperator('X8')
    ).astype(dtype)
    return [first, second]


def _builtin_gradient_circuit():  # pylint: disable=too-many-statements
    """Exercise every intrinsic derivative lowered by the adjoint tile builder."""
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += G.X.on(1, 9)
    circuit += G.RX({'shared': 1.7, 'rx': -0.3}).on(0, 8)
    circuit += G.RY('ry').on(1, 9)
    circuit += G.RZ('rz').on(2, 8)
    circuit += G.Rxx('rxx').on([0, 3], 9)
    circuit += G.Ryy('ryy').on([1, 4], 8)
    circuit += G.Rzz('rzz').on([2, 5], 9)
    circuit += G.Rxy('rxy').on([3, 6], 8)
    circuit += G.Rxz('rxz').on([4, 7], 9)
    circuit += G.Ryz('ryz').on([0, 5], 8)
    circuit += G.SWAPalpha('swap_alpha').on([1, 6], 9)
    circuit += G.Givens('givens').on([2, 7], 8)
    circuit += G.RotPauliString('XYZIZ', 'rps').on([0, 2, 4, 6, 7], 9)
    circuit += G.PhaseShift({'phase': 0.8, 'shared': -0.2}).on(5, 8)
    circuit += G.GlobalPhase('global_phase').on(0, 9)
    circuit += G.U3('u3_theta', 'u3_phi', 'u3_lambda').on(6, 8)
    circuit += G.Rn('rn_alpha', 'rn_beta', 'rn_gamma').on(7, 9)
    circuit += G.FSim('fsim_theta', 'fsim_phi').on([3, 5], 8)
    circuit += G.SX.on(4)
    circuit += G.H.on(7, 9)
    return circuit


def _known_gate_circuit():  # pylint: disable=too-many-statements
    """Build a circuit covering every gate supported by the tile builder."""
    circuit = Circuit()
    circuit += G.H.on(0, 11)
    circuit += G.X.on(8, 10)
    circuit += G.Y.on(1)
    circuit += G.Z.on(9, 11)
    circuit += G.S.on(2)
    circuit += G.S.hermitian().on(3, 10)
    circuit += G.T.on(4)
    circuit += G.T.hermitian().on(5, 11)
    circuit += G.SX.on(6)
    circuit += G.SX.hermitian().on(7, 10)
    circuit += G.H.on(10)
    circuit += G.RX('rx').on(0, 10)
    circuit += G.RY('ry').on(8, 11)
    circuit += G.RZ('rz').on(9, 10)
    circuit += G.Rxx('rxx').on([0, 8], 11)
    circuit += G.Ryy('ryy').on([1, 9], 10)
    circuit += G.Rzz('rzz').on([2, 7], 11)
    circuit += G.Rxy('rxy').on([3, 6], 10)
    circuit += G.Rxz('rxz').on([4, 9], 11)
    circuit += G.Ryz('ryz').on([5, 7], 10)
    circuit += G.SWAP.on([0, 9], 10)
    circuit += G.ISWAP.on([1, 8], 11)
    circuit += G.ISWAP.hermitian().on([2, 7], 10)
    circuit += G.SWAPalpha('swap_alpha').on([3, 6], 11)
    circuit += G.Givens('givens').on([4, 5], 10)
    circuit += G.GroupedPauli('XYZIZ').on([0, 2, 4, 6, 8], 11)
    circuit += G.RotPauliString('XYZIZ', 'rps').on([1, 3, 5, 7, 9], 10)
    circuit += G.PhaseShift('phase').on(9, 11)
    circuit += G.GlobalPhase('global_phase').on(0, 10)
    circuit += G.U3('u3_theta', 'u3_phi', 'u3_lambda').on(8, 11)
    circuit += G.Rn('rn_alpha', 'rn_beta', 'rn_gamma').on(7, 10)
    circuit += G.FSim('fsim_theta', 'fsim_phi').on([6, 9], 11)
    circuit += G.UnivMathGate('custom_1q', _unitary(2, 101)).on(5, 10)
    circuit += G.UnivMathGate('custom_3q', _unitary(8, 102)).on([8, 3, 0], 11)
    circuit += G.H.on(11)
    return circuit


def _parameters():
    """Return deterministic values for every parameter in the known-gate circuit."""
    names = [
        'rx',
        'ry',
        'rz',
        'rxx',
        'ryy',
        'rzz',
        'rxy',
        'rxz',
        'ryz',
        'swap_alpha',
        'givens',
        'rps',
        'phase',
        'global_phase',
        'u3_theta',
        'u3_phi',
        'u3_lambda',
        'rn_alpha',
        'rn_beta',
        'rn_gamma',
        'fsim_theta',
        'fsim_phi',
    ]
    return dict(zip(names, np.linspace(-1.7, 2.1, len(names))))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_circuit_tiling_matches_sequential_known_gate_paths(dtype):
    """
    Feature: Circuit-tiled execution of built-in and dense unitary gates.
    Description: Apply the same mixed gate circuit through tiled and sequential execution for both precisions.
    Expectation: The final tiled state matches the sequential reference within the precision tolerance.
    """
    n_qubits = _OPENMP_QUBIT_THRESHOLD
    circuit = _known_gate_circuit()
    parameters = _parameters()
    state = _normalized_state(n_qubits, 20260713)

    reference = Simulator('mqvector', n_qubits, dtype=dtype)
    reference.set_qs(state)
    for gate in circuit:
        reference.apply_gate(gate, parameters)

    tiled = Simulator('mqvector', n_qubits, dtype=dtype)
    tiled.set_qs(state)
    tiled.apply_circuit(circuit, parameters)

    atol = 3e-5 if dtype == mq.complex64 else 2e-12
    assert np.allclose(tiled.get_qs(), reference.get_qs(), atol=atol, rtol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_adjoint_circuit_tilings_match_legacy_for_builtin_derivatives(dtype):
    """
    Feature: Circuit-tiled adjoint differentiation for built-in gates.
    Description: Compare tiled and legacy expectation gradients across every built-in derivative kernel.
    Expectation: Values and gradients match the legacy path within the precision tolerance.
    """
    circuit = _builtin_gradient_circuit()
    parameters = np.linspace(-1.3, 1.7, len(circuit.params_name))
    reference_state = _normalized_state(_SEQUENTIAL_QUBITS, 20260717)
    hamiltonians = _gradient_hamiltonians(dtype)

    reference = Simulator('mqvector', _SEQUENTIAL_QUBITS, dtype=dtype)
    reference.set_qs(reference_state)
    reference_value, reference_gradient = reference.get_expectation_with_grad(hamiltonians, circuit)(parameters)

    tiled = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD, dtype=dtype)
    tiled.set_qs(_idle_high_qubit(reference_state))
    tiled_value, tiled_gradient = tiled.get_expectation_with_grad(hamiltonians, circuit)(parameters)

    atol = 8e-5 if dtype == mq.complex64 else 3e-11
    assert np.allclose(tiled_value, reference_value, atol=atol, rtol=atol)
    assert np.allclose(tiled_gradient, reference_gradient, atol=atol, rtol=atol)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_single_gate_adjoint_tile_matches_legacy_path():
    """
    Feature: Single-gate adjoint tile execution.
    Description: Differentiate one controlled parameterized gate above and below the tiling threshold.
    Expectation: Tiled values and gradients match the legacy reference.
    """
    circuit = Circuit(G.RX({'a': 1.25, 'b': -0.4}).on(3, 9))
    parameters = np.array([0.37, -0.61])
    reference_state = _normalized_state(_SEQUENTIAL_QUBITS, 20260718)
    hamiltonian = _gradient_hamiltonians(mq.complex128)[0]

    reference = Simulator('mqvector', _SEQUENTIAL_QUBITS)
    reference.set_qs(reference_state)
    expected = reference.get_expectation_with_grad(hamiltonian, circuit)(parameters)

    tiled = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    tiled.set_qs(_idle_high_qubit(reference_state))
    actual = tiled.get_expectation_with_grad(hamiltonian, circuit)(parameters)

    assert np.allclose(actual[0], expected[0], atol=2e-12, rtol=2e-12)
    assert np.allclose(actual[1], expected[1], atol=2e-12, rtol=2e-12)


def _apply_matrix_reference(state, matrix, objects, controls):
    """Apply a dense matrix to selected object qubits without simulator tiling."""
    result = state.copy()
    n_qubits = state.size.bit_length() - 1
    object_mask = sum(1 << qubit for qubit in objects)
    control_mask = sum(1 << qubit for qubit in controls)
    offsets = []
    for basis in range(1 << len(objects)):
        offset = 0
        for bit, qubit in enumerate(objects):
            if basis & (1 << bit):
                offset |= 1 << qubit
        offsets.append(offset)
    for base in range(1 << n_qubits):
        if base & object_mask or base & control_mask != control_mask:
            continue
        indices = [base | offset for offset in offsets]
        result[indices] = matrix @ state[indices]
    return result


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("dtype", [mq.complex64, mq.complex128])
def test_dense_matrix_width_boundary_and_object_order(dtype):
    """
    Feature: Dense-matrix tile lowering across kernel-width boundaries.
    Description: Apply one- through six-qubit matrices with deliberately non-sorted object qubits.
    Expectation: Each simulator result matches a direct reference while preserving object-qubit order.
    """
    n_qubits = 11
    state = _normalized_state(n_qubits, 20260714)
    object_order = [7, 1, 5, 0, 8, 3]
    controls = [10]
    atol = 5e-5 if dtype == mq.complex64 else 2e-12

    for n_objects in range(1, 7):
        objects = object_order[:n_objects]
        matrix = _unitary(1 << n_objects, 200 + n_objects)
        gate = G.UnivMathGate(f'dense_{n_objects}', matrix).on(objects, controls)
        expected = _apply_matrix_reference(state, matrix, objects, controls)

        simulator = Simulator('mqvector', n_qubits, dtype=dtype)
        simulator.set_qs(state)
        simulator.apply_gate(gate)

        assert np.allclose(simulator.get_qs(), expected, atol=atol, rtol=atol)


try:
    import numba  # noqa: F401

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _custom_one_matrix(theta):
    cosine = np.cos(theta / 2)
    minus_i_sine = -1j * np.sin(theta / 2)
    return np.array([[cosine, minus_i_sine], [minus_i_sine, cosine]], dtype=np.complex128)


def _custom_one_diff(theta):
    diagonal = -0.5 * np.sin(theta / 2)
    off_diagonal = -0.5j * np.cos(theta / 2)
    return np.array([[diagonal, off_diagonal], [off_diagonal, diagonal]], dtype=np.complex128)


def _custom_two_matrix(theta, phi):
    cosine = np.cos(theta)
    minus_i_sine = -1j * np.sin(theta)
    return np.array(
        [
            [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, cosine, minus_i_sine, 0.0j],
            [0.0j, minus_i_sine, cosine, 0.0j],
            [0.0j, 0.0j, 0.0j, np.exp(-1j * phi)],
        ],
        dtype=np.complex128,
    )


def _custom_two_theta_diff(theta, _phi):  # pylint: disable=invalid-name
    minus_sine = -np.sin(theta)
    minus_i_cosine = -1j * np.cos(theta)
    return np.array(
        [
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, minus_sine, minus_i_cosine, 0.0j],
            [0.0j, minus_i_cosine, minus_sine, 0.0j],
            [0.0j, 0.0j, 0.0j, 0.0j],
        ],
        dtype=np.complex128,
    )


def _custom_two_phi_diff(_theta, phi):  # pylint: disable=invalid-name
    return np.array(
        [
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, 0.0j, 0.0j, -1j * np.exp(-1j * phi)],
        ],
        dtype=np.complex128,
    )


def _parameterized_custom_gates():
    one = G.gene_univ_parameterized_gate('custom_one', _custom_one_matrix, _custom_one_diff)
    two = G.gene_univ_two_params_gate('custom_two', _custom_two_matrix, _custom_two_theta_diff, _custom_two_phi_diff)
    return one, two


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_parameterized_custom_gates_inside_tile():
    """
    Feature: Parameterized custom gates inside circuit tiles.
    Description: Execute one- and two-parameter callback gates through tiled and sequential paths.
    Expectation: Callback matrices are resolved correctly and both final states match.
    """
    custom_one, custom_two = _parameterized_custom_gates()
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += custom_one('a').on(1, 10)
    circuit += custom_two('b', 'c').on([2, 7], 9)
    circuit += G.X.on(8)
    parameters = {'a': 0.37, 'b': -0.41, 'c': 0.83}
    state = _normalized_state(_OPENMP_QUBIT_THRESHOLD, 20260715)

    reference = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    reference.set_qs(state)
    for gate in circuit:
        reference.apply_gate(gate, parameters)

    tiled = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    tiled.set_qs(state)
    tiled.apply_circuit(circuit, parameters)

    assert np.allclose(tiled.get_qs(), reference.get_qs(), atol=2e-12, rtol=2e-12)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.skipif(not _HAS_NUMBA, reason='Numba is not installed')
def test_parameterized_custom_gate_derivatives_inside_adjoint_tile():
    """
    Feature: Parameterized custom-gate derivatives inside adjoint tiles.
    Description: Compare tiled and legacy gradients for one- and two-parameter callback gates.
    Expectation: Tiled expectation values and gradients match the legacy reference.
    """
    custom_one, custom_two = _parameterized_custom_gates()
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += custom_one({'shared': 0.5, 'a': -1.3}).on(1, 9)
    circuit += custom_two('b', {'shared': -0.4, 'c': 0.7}).on([2, 7], 8)
    circuit += G.X.on(6)
    parameters = np.linspace(-0.9, 1.1, len(circuit.params_name))
    reference_state = _normalized_state(_SEQUENTIAL_QUBITS, 20260719)
    hamiltonians = _gradient_hamiltonians(mq.complex128)

    reference = Simulator('mqvector', _SEQUENTIAL_QUBITS)
    reference.set_qs(reference_state)
    expected = reference.get_expectation_with_grad(hamiltonians, circuit)(parameters)

    tiled = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    tiled.set_qs(_idle_high_qubit(reference_state))
    actual = tiled.get_expectation_with_grad(hamiltonians, circuit)(parameters)

    assert np.allclose(actual[0], expected[0], atol=3e-11, rtol=3e-11)
    assert np.allclose(actual[1], expected[1], atol=3e-11, rtol=3e-11)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_non_hermitian_adjoint_tiles_match_legacy_path():
    """
    Feature: Non-Hermitian adjoint circuit tiling.
    Description: Evaluate distinct left and right parameterized circuits with complex Hamiltonians.
    Expectation: Tiled values and complex gradients match the legacy left/right sweep.
    """
    right_circuit = Circuit()
    right_circuit += G.H.on(0)
    right_circuit += G.RX({'shared': 0.6, 'right_rx': -1.1}).on(1, 9)
    right_circuit += G.FSim('right_theta', 'right_phi').on([2, 6], 8)
    right_circuit += G.S.on(7)
    left_circuit = Circuit()
    left_circuit += G.X.on(3)
    left_circuit += G.RY({'shared': -0.35, 'left_ry': 0.8}).on(4, 8)
    left_circuit += G.Rn('left_alpha', 'left_beta', 'left_gamma').on(5, 9)
    left_circuit += G.T.on(0)
    parameter_count = len(set(right_circuit.params_name) | set(left_circuit.params_name))
    parameters = np.linspace(-1.2, 1.4, parameter_count)
    right_state = _normalized_state(_SEQUENTIAL_QUBITS, 20260720)
    left_state = _normalized_state(_SEQUENTIAL_QUBITS, 20260721)
    hamiltonians = _gradient_hamiltonians(mq.complex128)

    reference_right = Simulator('mqvector', _SEQUENTIAL_QUBITS)
    reference_right.set_qs(right_state)
    reference_left = Simulator('mqvector', _SEQUENTIAL_QUBITS)
    reference_left.set_qs(left_state)
    expected = reference_right.get_expectation_with_grad(hamiltonians, right_circuit, left_circuit, reference_left)(
        parameters
    )

    tiled_right = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    tiled_right.set_qs(_idle_high_qubit(right_state))
    tiled_left = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    tiled_left.set_qs(_idle_high_qubit(left_state))
    actual = tiled_right.get_expectation_with_grad(hamiltonians, right_circuit, left_circuit, tiled_left)(parameters)

    assert np.allclose(actual[0], expected[0], atol=3e-11, rtol=3e-11)
    assert np.allclose(actual[1], expected[1], atol=3e-11, rtol=3e-11)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qram_adjoint_tiles_match_legacy_path():
    """
    Feature: Batched QRAM adjoint tile execution.
    Description: Evaluate two input states with one prepared tiled program and the legacy backend.
    Expectation: Every batched expectation value and gradient matches the legacy result.
    """
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += G.RY('a').on(2, 9)
    circuit += G.Rxx({'b': 0.7, 'c': -0.2}).on([3, 6], 8)
    circuit += G.S.on(7)
    parameters = np.linspace(-0.4, 0.9, len(circuit.params_name))
    reference_states = np.stack(
        (
            _normalized_state(_SEQUENTIAL_QUBITS, 20260722),
            _normalized_state(_SEQUENTIAL_QUBITS, 20260723),
        )
    )
    tiled_states = np.concatenate((reference_states, np.zeros_like(reference_states)), axis=1)
    hamiltonians = _gradient_hamiltonians(mq.complex128)

    reference = Simulator('mqvector', _SEQUENTIAL_QUBITS)
    expected = reference.backend.sim.qram_expectation_with_grad(
        [ham.get_cpp_obj() for ham in hamiltonians],
        circuit.get_cpp_obj(),
        circuit.get_cpp_obj(hermitian=True),
        reference_states,
        parameters,
        circuit.params_name,
        1,
        len(hamiltonians),
    )

    tiled = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    actual = tiled.backend.sim.qram_expectation_with_grad(
        [ham.get_cpp_obj() for ham in hamiltonians],
        circuit.get_cpp_obj(),
        circuit.get_cpp_obj(hermitian=True),
        tiled_states,
        parameters,
        circuit.params_name,
        1,
        len(hamiltonians),
    )

    assert np.allclose(actual, expected, atol=3e-11, rtol=3e-11)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_identity_pauli_strings_form_a_noop_tile():
    """
    Feature: Identity-only Pauli-string tile planning.
    Description: Apply a circuit containing only grouped identity Pauli strings.
    Expectation: Planning and execution leave the state exactly unchanged.
    """
    state = _normalized_state(_OPENMP_QUBIT_THRESHOLD, 20260716)
    circuit = Circuit()
    circuit += G.GroupedPauli('III').on([0, 4, 9])
    circuit += G.GroupedPauli('II').on([2, 7])

    simulator = Simulator('mqvector', _OPENMP_QUBIT_THRESHOLD)
    simulator.set_qs(state)
    expected = simulator.get_qs().copy()
    simulator.apply_circuit(circuit)

    assert np.array_equal(simulator.get_qs(), expected)
