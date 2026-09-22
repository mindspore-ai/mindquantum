# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hardware conformance of Ascend Vector/Cube execution and layout invariants."""

import ctypes
import gc
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core import gates as G  # noqa: N812
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.simulator.mqvector_npu import MQVectorNPU
from mindquantum.utils import random_circuit

pytestmark = pytest.mark.skipif(find_spec('mindquantum._mq_vector_npu') is None, reason='Ascend extension not built')


def _state(n_qubits, seed=831):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=1 << n_qubits) + 1j * rng.normal(size=1 << n_qubits)
    return values / np.linalg.norm(values)


def _numpy_gate(state, matrix, targets, controls=()):
    """Independent little-endian embedding, including target argument order."""
    indices = np.arange(state.size, dtype=np.uint64)
    target_mask = sum(1 << bit for bit in targets)
    control_mask = sum(1 << bit for bit in controls)
    bases = indices[((indices & target_mask) == 0) & ((indices & control_mask) == control_mask)]
    offsets = [
        sum(((basis >> bit) & 1) << target for bit, target in enumerate(targets)) for basis in range(len(matrix))
    ]
    groups = bases[:, None] | np.array(offsets, dtype=np.uint64)
    result = state.copy()
    result[groups] = state[groups] @ matrix.T
    return result


@pytest.mark.parametrize('n_qubits', [0, 1, 2, 3, 7, 12, 17])
def test_reset_roundtrip_and_copy(n_qubits):
    sim = Simulator(MQVectorNPU(n_qubits, seed=17))
    expected = np.zeros(1 << n_qubits, dtype=np.complex64)
    expected[0] = 1
    np.testing.assert_array_equal(sim.get_qs(), expected)
    values = _state(n_qubits)
    sim.set_qs(values)
    saved = sim.copy()
    sim.reset()
    np.testing.assert_array_equal(sim.get_qs(), expected)
    np.testing.assert_allclose(saved.get_qs(), values, rtol=2e-7, atol=1e-8)
    assert saved.dtype == mq.complex64


def test_readback_snapshot_owns_storage_independent_of_device_and_views():
    sim = MQVectorNPU(7)
    values = _state(7)
    sim.set_qs(values)
    snapshot = sim.get_qs()
    view = snapshot[1::3]
    sim.reset()
    np.testing.assert_allclose(snapshot, values, rtol=2e-7, atol=1e-8)
    del sim, snapshot
    gc.collect()
    np.testing.assert_allclose(view, values[1::3], rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize('target', [0, 1, 2, 6, 11, 15])
@pytest.mark.parametrize('controls', [(), (7,), (7, 14)])
def test_asymmetric_matrix_across_dma_and_external_controls(target, controls):
    sim = MQVectorNPU(16)
    values = _state(16)
    # Nonreal, nonsymmetric entries detect conjugation and row/column mistakes.
    gate = G.U3(0.371, -0.827, 1.219).on(target, list(controls))
    expected = _numpy_gate(values, gate.matrix(), [target], controls)
    sim.set_qs(values)
    sim.apply_gate(gate)
    actual = sim.get_qs()
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-8)
    assert np.linalg.norm(actual - expected) < 2e-7


def test_strided_pair_spans_and_masked_tails():
    # q=3 crosses the 255-repeat hardware boundary; q=7..10 change which
    # dimension is vectorized. Low controls fragment otherwise aligned runs.
    values = _state(16, seed=91)
    sim = MQVectorNPU(16)
    for target in range(3, 13):
        for controls in [[], [0], [14], [0, 1, 2, 13, 14, 15]]:
            gate = G.U3(-0.73, 1.219, 0.491).on(target, controls)
            sim.set_qs(values)
            sim.apply_gate(gate)
            expected = _numpy_gate(values, gate.matrix(), [target], controls)
            assert np.linalg.norm(sim.get_qs() - expected) < 2e-7


@pytest.mark.parametrize(
    'gate',
    [
        G.H,
        G.X,
        G.Y,
        G.Z,
        G.S,
        G.S.hermitian(),
        G.T,
        G.T.hermitian(),
        G.SX,
        G.SX.hermitian(),
        G.RX(0.327),
        G.RY(-1.137),
        G.RZ(2.213),
        G.PhaseShift(-0.715),
        G.GlobalPhase(0.891),
        G.U3(0.19, -0.82, 1.23),
        G.Rn(0.71, -0.53, 1.19),
        G.SWAP,
        G.ISWAP,
        G.ISWAP.hermitian(),
        G.SWAPalpha(0.393),
        G.Givens(-0.219),
        G.Rxx(0.39),
        G.Rxy(-1.17),
        G.Rxz(0.73),
        G.Ryy(-0.92),
        G.Ryz(1.38),
        G.Rzz(0.57),
        G.FSim(0.59, -1.21),
    ],
)
def test_builtin_gates_and_inverse(gate):
    n_qubits = 5
    targets = [2] if gate.n_qubits == 1 else [2, 0]
    controls = [1, 4]
    applied = gate.on(targets, controls)
    values = _state(n_qubits)
    expected = _numpy_gate(values, gate.matrix(), targets, controls)
    sim = MQVectorNPU(n_qubits)
    sim.set_qs(values)
    sim.apply_gate(applied)
    np.testing.assert_allclose(sim.get_qs(), expected, rtol=2e-6, atol=1e-7)
    sim.apply_gate(applied.hermitian())
    np.testing.assert_allclose(sim.get_qs(), values, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize('targets', [[0, 13], [13, 0], [7, 12]])
def test_dense_two_qubit_matrix(targets):
    rng = np.random.default_rng(772)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    gate = G.UnivMathGate('asymmetric', matrix).on(targets, [14])
    values = _state(15)
    sim = MQVectorNPU(15)
    sim.set_qs(values)
    sim.apply_gate(gate)
    expected = _numpy_gate(values, matrix, targets, [14])
    assert np.linalg.norm(sim.get_qs() - expected) < 2e-7


@pytest.mark.parametrize('tile_qubits,fusion_qubits', [(8, 4), (10, 6), (12, 6), (12, 8), (12, 9)])
def test_fusion_matches_sequential_complex128_and_preserves_norm(tile_qubits, fusion_qubits):
    n_qubits = 15
    rng = np.random.default_rng(20931)
    circuit = Circuit()
    for _ in range(90):
        a, b, c = map(int, rng.choice(n_qubits, 3, replace=False))
        circuit += G.RX(float(rng.uniform(-3, 3))).on(a)
        circuit += G.RY(float(rng.uniform(-3, 3))).on(b, c)
        circuit += G.Rzz(float(rng.uniform(-3, 3))).on([a, c])
        circuit += G.X.on(c, a)
    # Exercise external Z parity independently from external controls.
    circuit += G.GroupedPauli('XYZIZ').on([0, 2, 11, 3, 14], 13)
    circuit += G.RotPauliString('XYZIZ', -0.273).on([1, 4, 12, 5, 14], 13)
    circuit += G.Rxz(0.817).on([3, 14], 13)
    circuit += G.GlobalPhase(-0.439).on(4, 14)
    values = _state(n_qubits)
    reference = Simulator('mqvector', n_qubits, dtype=mq.complex128)
    reference.set_qs(values)
    for gate in circuit:
        reference.apply_gate(gate)
    sim = MQVectorNPU(n_qubits, tile_qubits=tile_qubits, fusion_qubits=fusion_qubits)
    sim.set_qs(values)
    compiled = sim.compile(circuit)
    assert compiled.tile_count < len(circuit)
    sim.apply_compiled(compiled)
    actual = sim.get_qs().astype(np.complex128)
    assert np.linalg.norm(actual - reference.get_qs()) < 1e-5
    assert abs(np.vdot(actual, actual).real - 1) < 1e-5
    sim.apply_circuit(circuit.hermitian())
    assert np.linalg.norm(sim.get_qs() - values) < 2e-5


@pytest.mark.parametrize('cube', [False, True])
def test_compiled_snapshot_reuse_lifetime_and_threads(cube):
    sim = MQVectorNPU(13, cube_min_gates=32 if cube else 0)
    circuit = Circuit([G.H.on(2), G.RY('angle').on(11, 3), G.Rxx(0.871).on([0, 8])])
    if cube:
        circuit = Circuit([G.U3('angle', -0.817, 1.317).on(i % 6) for i in range(40)]) + circuit
    program = sim.compile(circuit, {'angle': 0.413})
    assert bool(program.cube_tile_count) == cube
    circuit += G.X.on(12)
    values = _state(13)
    reference = Simulator('mqvector', 13)
    reference.set_qs(values)
    original = circuit[:-1].apply_value({'angle': 0.413})
    for _ in range(3):
        for gate in original:
            reference.apply_gate(gate)
    first, second = sim.copy(), sim.copy()
    del sim

    def execute(backend):
        backend.set_qs(values)
        backend.apply_compiled(program, repetitions=3)
        return backend.get_qs()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(execute, [first, second]))
    for output in outputs:
        assert np.linalg.norm(output - reference.get_qs()) < 1e-6


@pytest.mark.parametrize('width', [4, 5, 6])
@pytest.mark.parametrize('n_qubits', [12, 17])
def test_cube_fusion_complex_phase_controls_and_replay(width, n_qubits):
    # Both original planes must be consumed before the realified GEMM writes.
    # These asymmetric coefficients also distinguish FP32 from HF32/FP16.
    circuit = Circuit()
    for i in range(40):
        target = i % width
        circuit += G.U3(0.371 + i / 137, -0.829, 1.219).on(target, (target + 1) % width)
        if i % 3 == 0:
            circuit += G.Rxy(-0.419).on([target, (target + 2) % width])
    circuit += G.GlobalPhase(0.139).on(0)
    circuit += G.RotPauliString('XYZ', 'theta').on([width - 1, 1, 0])
    values = _state(n_qubits, seed=219)
    reference = Simulator('mqvector', n_qubits, dtype=mq.complex128)
    reference.set_qs(values)
    bound = circuit.apply_value({'theta': 0.715})
    for _ in range(3):
        for gate in bound:
            reference.apply_gate(gate)
    sim = MQVectorNPU(n_qubits, cube_min_gates=32)
    sim.set_qs(values)
    executable = sim.compile(circuit, {'theta': 0.715})
    # One contraction followed by exact restoration to the public basis.
    assert executable.cube_tile_count == 1
    assert executable.tile_count == 2
    circuit += G.X.on(n_qubits - 1)  # compiled operators are snapshots
    sim.apply_compiled(executable, repetitions=3)
    actual = sim.get_qs().astype(np.complex128)
    assert np.linalg.norm(actual - reference.get_qs()) < 7e-7
    assert abs(np.vdot(actual, actual).real - 1) < 1e-6
    sim.apply_compiled(sim.compile(bound.hermitian()), repetitions=3)
    assert np.linalg.norm(sim.get_qs() - values) < 1e-6


def test_mixed_cube_vector_ordering_and_explicit_vector_policy():
    local = Circuit([G.U3(0.17 + i / 127, -0.37, 0.29).on(i % 6) for i in range(48)])
    circuit = local + Circuit([G.X.on(0, 13), G.RY(-0.73).on(13, 0)]) + local.hermitian()
    values = _state(16)
    reference = Simulator('mqvector', 16, dtype=mq.complex128)
    reference.set_qs(values)
    for gate in circuit:
        reference.apply_gate(gate)
    for minimum in [0, 32]:
        sim = MQVectorNPU(16, cube_min_gates=minimum)
        executable = sim.compile(circuit)
        if minimum:
            assert 0 < executable.cube_tile_count < executable.tile_count
        else:
            assert executable.cube_tile_count == 0
        sim.set_qs(values)
        sim.apply_compiled(executable)
        assert np.linalg.norm(sim.get_qs() - reference.get_qs()) < 2e-6


def test_cube_rejection_before_state_mutation_and_selection_boundary():
    circuit = Circuit([G.RX(0.1 + i / 53).on(i % 4) for i in range(31)])
    sim = MQVectorNPU(12, cube_min_gates=32)
    assert sim.compile(circuit).cube_tile_count == 0
    circuit += G.RY(-0.31).on(3)
    assert sim.compile(circuit).cube_tile_count == 1
    assert MQVectorNPU(12).compile(circuit).cube_tile_count == 1
    assert MQVectorNPU(12, cube_min_gates=0).compile(circuit).cube_tile_count == 0
    small = MQVectorNPU(11, cube_min_gates=32)
    assert small.compile(circuit).cube_tile_count == 0
    sim.set_qs(_state(12))
    before = sim.get_qs()
    for bad_gate in [G.Measure().on(0), G.RX(np.nan).on(2)]:
        with pytest.raises(ValueError):
            sim.compile(circuit + Circuit([bad_gate]))
        np.testing.assert_array_equal(sim.get_qs(), before)
    with pytest.raises(ValueError, match='cube_min_gates'):
        MQVectorNPU(12, cube_min_gates=-1)


@pytest.mark.parametrize('n_qubits', [7, 12])
def test_caller_context_is_preserved_during_use_and_destruction(n_qubits):
    bootstrap = MQVectorNPU(0)
    del bootstrap
    acl = ctypes.CDLL('libascendcl.so')
    pointer = ctypes.c_void_p
    acl.aclrtCreateContext.argtypes = [ctypes.POINTER(pointer), ctypes.c_int]
    acl.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(pointer)]
    acl.aclrtDestroyContext.argtypes = [pointer]
    foreign = pointer()
    assert acl.aclrtCreateContext(ctypes.byref(foreign), 0) == 0

    def check_owner():
        current = pointer()
        assert acl.aclrtGetCurrentContext(ctypes.byref(current)) == 0
        assert current.value == foreign.value

    try:
        sim = MQVectorNPU(n_qubits, cube_min_gates=32)
        check_owner()
        program = sim.compile(Circuit([G.U3(0.4 + i / 101, -0.7, 1.2).on(4) for i in range(40)]))
        assert bool(program.cube_tile_count) == (n_qubits == 12)
        check_owner()
        sim.apply_compiled(program)
        check_owner()
        sim.set_qs(_state(n_qubits))
        check_owner()
        sim.get_qs()
        check_owner()
        copied = sim.copy()
        check_owner()
        sim.reset()
        check_owner()
        del sim, copied, program
        gc.collect()
        check_owner()
    finally:
        assert acl.aclrtDestroyContext(foreign) == 0


@pytest.mark.parametrize('n_qubits', [13, 17])
@pytest.mark.parametrize('seed', [4099, 65537])
def test_general_random_circuit_n_times_100(n_qubits, seed):
    circuit = random_circuit(n_qubits, n_qubits * 100, seed=seed)
    values = _state(n_qubits, seed=seed + 1)
    reference = Simulator('mqvector', n_qubits, dtype=mq.complex128)
    reference.set_qs(values)
    for gate in circuit:
        reference.apply_gate(gate)
    sim = MQVectorNPU(n_qubits)
    sim.set_qs(values)
    executable = sim.compile(circuit)
    assert 0 < executable.cube_tile_count <= executable.tile_count
    sim.apply_compiled(executable)
    actual = sim.get_qs().astype(np.complex128)
    assert np.linalg.norm(actual - reference.get_qs()) < 2e-5
    assert abs(np.vdot(actual, actual).real - 1) < 2e-5


@pytest.mark.parametrize('target', [0, 7])
def test_matrix_zero_row_and_delayed_first_nonzero_contribution(target):
    # A stale output row or initialization based only on the first real
    # coefficient fails these generic matrices, on Gather and strided paths.
    values = _state(13)
    sim = MQVectorNPU(13)
    for matrix in [np.array([[0, 0], [0.3j, -0.7]]), np.array([[0, 0.7j], [-0.2j, 0]])]:
        gate = G.UnivMathGate('sparse', matrix).on(target, 11)
        sim.set_qs(values)
        sim.apply_gate(gate)
        expected = _numpy_gate(values, matrix, [target], [11])
        assert np.linalg.norm(sim.get_qs() - expected) < 2e-7


@pytest.mark.parametrize('controls', [[], [7, 13]])
def test_complex_plane_dependencies_and_zero_rows_across_dense_operations(controls):
    # Consecutive operations reuse UB scratch. A plane must not read the other
    # plane's partial result or retain a prior row when its first terms are zero.
    matrix = np.array([[0, 0, 0, 0], [0, 0, 0.43j, -0.71], [0.23, -0.81j, 0, 0], [0, 0.91 - 0.31j, 0.29j, 0.27]])
    rng = np.random.default_rng(331)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    values = _state(15, seed=337)
    expected = values.copy()
    circuit = Circuit()
    for targets in [[0, 14], [14, 0], [0, 14]]:
        for operator in [unitary, matrix]:
            circuit += G.UnivMathGate('dense', operator).on(targets, controls)
            expected = _numpy_gate(expected, operator, targets, controls)
    sim = MQVectorNPU(15)
    sim.set_qs(values)
    sim.apply_circuit(circuit)
    assert np.linalg.norm(sim.get_qs() - expected) < 3e-7


@pytest.mark.parametrize(
    'targets,controls,instructions',
    [([0, 1, 14], [], 2), ([6, 9, 14], [12, 13], 2), ([0, 1, 2], list(range(3, 15)), 1)],
)
def test_disjoint_matrix_blocks_share_indices_and_preserve_predicates(targets, controls, instructions):
    # Low targets need Gather; high local targets permit PairSpan. With many
    # controls each block has only a few lanes: padding must not overwrite the
    # inactive tail. External Z creates two predicates, never one merged block.
    values = _state(15, seed=787)
    sim = MQVectorNPU(15, cube_min_gates=0)
    for gate in [G.GroupedPauli('YXZ'), G.RotPauliString('YXZ', -0.417)]:
        applied = gate.on(targets, controls)
        executable = sim.compile(Circuit([applied]))
        assert executable.instruction_count == instructions
        sim.set_qs(values)
        sim.apply_compiled(executable)
        expected = _numpy_gate(values, gate.matrix(), targets, controls)
        actual = sim.get_qs()
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-8)
        assert np.linalg.norm(actual - expected) < 2e-7


@pytest.mark.parametrize('tile_qubits,fusion_qubits', [(6, 3), (9, 6), (12, 6)])
def test_compact_support_spectators_and_diagonal_predicates(tile_qubits, fusion_qubits):
    # Unused bits inside a DMA tile are independent copies, not controls.
    # Changing the tile width moves controls/Z operands across its boundary.
    circuit = Circuit()
    for angle in [0.0, -0.719, 1.231]:
        circuit += G.U3(angle, -0.327, 1.173).on(3, 1)
        circuit += G.RZ(0.39).on(0)
        circuit += G.PhaseShift(angle).on(12, 1)
        circuit += G.GlobalPhase(-0.27).on(2)
        circuit += G.Z.on(14)
        circuit += G.Rzz(-0.83).on([0, 14], 11)
        circuit += G.Rxz(angle).on([3, 14], 11)
        circuit += G.FSim(-0.49, angle).on([12, 3], 1)
        circuit += G.Givens(angle).on([3, 12])
        circuit += G.GroupedPauli('YZX').on([3, 14, 12], [1, 11])
    values = _state(15, seed=277)
    reference = Simulator('mqvector', 15, dtype=mq.complex128)
    reference.set_qs(values)
    for gate in circuit:
        reference.apply_gate(gate)
    sim = MQVectorNPU(15, tile_qubits=tile_qubits, fusion_qubits=fusion_qubits)
    sim.set_qs(values)
    sim.apply_circuit(circuit)
    assert np.linalg.norm(sim.get_qs() - reference.get_qs()) < 2e-6


def test_index_reuse_does_not_alias_coefficients_or_executable_storage():
    # Repeat layouts across tiles with different coefficients and target order.
    # Retain two executables, then destroy the first; neither may borrow storage
    # from the other or from a process-global compilation cache.
    rng = np.random.default_rng(9967)
    matrices = [np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0] for _ in range(2)]
    circuits = []
    for matrix in matrices:
        circuit = Circuit()
        for shift in range(3):
            for target in range(15):
                circuit += G.U3(0.19 + shift * 0.31, -0.73 + target / 11, 1.217).on(target)
            circuit += G.UnivMathGate('ordered', matrix).on([13, 0], [2, 11])
            circuit += G.UnivMathGate('reversed', matrix).on([0, 13], [2, 11])
        circuits.append(circuit)
    sim = MQVectorNPU(15)
    values = _state(15, seed=9968)
    first = sim.compile(circuits[0])
    second = sim.compile(circuits[1])
    assert first.tile_count > 1
    for executable, circuit in zip([first, second], circuits):
        sim.set_qs(values)
        sim.apply_compiled(executable)
        reference = Simulator('mqvector', 15, dtype=mq.complex128)
        reference.set_qs(values)
        for gate in circuit:
            reference.apply_gate(gate)
        assert np.linalg.norm(sim.get_qs() - reference.get_qs()) < 3e-6
    expected = sim.get_qs()
    del first, executable
    gc.collect()
    sim.set_qs(values)
    sim.apply_compiled(second)
    np.testing.assert_array_equal(sim.get_qs(), expected)


@pytest.mark.parametrize('bad_gate', [G.Measure(), G.DepolarizingChannel(0.2)])
def test_stochastic_circuit_cannot_be_frozen_as_deterministic_executable(bad_gate):
    sim = MQVectorNPU(4)
    sim.set_qs(_state(4))
    before = sim.get_qs()
    circuit = Circuit([G.H.on(0), bad_gate.on(1)])
    with pytest.raises(ValueError, match='without measurement or noise'):
        sim.compile(circuit)
    np.testing.assert_array_equal(sim.get_qs(), before)


def test_validation_and_empty_circuit():
    sim = MQVectorNPU(3)
    sim.apply_circuit(Circuit())
    before = sim.get_qs()
    for state in [np.zeros(8), np.full(8, np.nan), np.full(8, np.inf), np.ones((2, 4)), np.ones(7)]:
        with pytest.raises(ValueError):
            sim.set_qs(state)
        np.testing.assert_array_equal(sim.get_qs(), before)
    with pytest.raises(ValueError, match='parameter'):
        sim.apply_gate(G.RX('unbound').on(0))
    with pytest.raises(ValueError, match='finite'):
        sim.apply_gate(G.RX(np.nan).on(0))
    with pytest.raises(ValueError, match='six target'):
        MQVectorNPU(7).apply_gate(G.UnivMathGate('large', np.eye(128)).on(list(range(7))))
    with pytest.raises(TypeError, match='complex64'):
        MQVectorNPU(3, dtype=mq.complex128)
    with pytest.raises(ValueError, match='n_qubits'):
        MQVectorNPU(33)
    with pytest.raises(ValueError, match='tile_qubits'):
        MQVectorNPU(3, tile_qubits=13)
    np.testing.assert_array_equal(sim.get_qs(), before)


def _cube_region(targets):
    circuit = Circuit()
    for i in range(41):
        target = targets[i % len(targets)]
        control = targets[(i + 1) % len(targets)]
        circuit += G.U3(0.371 + i / 83, -0.827, 1.219).on(target, control if i % 3 else [])
    circuit += G.FSim(0.712, -0.913).on([targets[-1], targets[0]])
    circuit += G.Rzz(-0.497).on([targets[-2], targets[1]])
    circuit += G.RX('theta').on(targets[-1])
    return circuit


def _cube_reference(circuit, values, n_qubits, repetitions=1):
    reference = Simulator('mqvector', n_qubits, dtype=mq.complex128)
    reference.set_qs(values)
    for _ in range(repetitions):
        for gate in circuit:
            reference.apply_gate(gate)
    return reference.get_qs()


@pytest.mark.parametrize('targets', [[0, 2, 5, 10], [1, 4, 9, 14, 17], [0, 1, 6, 9, 13, 16]])
def test_cube_noncontiguous_support_and_compact_synthesis(targets):
    values = _state(18, seed=592)
    circuit = _cube_region(targets)
    bound = circuit.apply_value({'theta': -0.674})
    expected = _cube_reference(bound, values, 18, repetitions=3)
    sim = MQVectorNPU(18)
    sim.set_qs(values)
    program = sim.compile(circuit, {'theta': -0.674})
    assert program.cube_tile_count == 1
    assert program.tile_count == 2
    sim.apply_compiled(program, repetitions=3)
    assert np.linalg.norm(sim.get_qs() - expected) < 2e-6
    sim.apply_compiled(sim.compile(bound.hermitian()), repetitions=3)
    assert np.linalg.norm(sim.get_qs() - values) < 2e-6


def test_cube_control_dependencies_and_wide_pauli_boundaries():
    n_qubits = 17
    values = _state(n_qubits, seed=592)
    prefix = _cube_region([0, 3, 7, 11]).apply_value({'theta': 0.57})
    circuit = Circuit()
    for control in (14, 15):
        circuit += prefix
        circuit += G.RY(0.371).on(0, control)
        circuit += G.X.on(control, 0)
        circuit += prefix.hermitian()
    # Z operands and controls also consume dense support. A seven-operand
    # string must stay on Vector instead of silently dropping its predicates.
    circuit += G.RotPauliString('XYZIZZZ', 0.817).on([1, 3, 5, 8, 10, 13, 16], [15])
    circuit += prefix
    expected = _cube_reference(circuit, values, n_qubits)
    sim = MQVectorNPU(n_qubits)
    program = sim.compile(circuit)
    assert 0 < program.cube_tile_count < program.tile_count
    sim.set_qs(values)
    sim.apply_compiled(program)
    assert np.linalg.norm(sim.get_qs() - expected) < 3e-6


@pytest.mark.parametrize('n_qubits', [17, 21])
def test_cube_executable_shared_by_independent_streams_and_copies(n_qubits):
    values = _state(n_qubits, seed=592)
    circuit = _cube_region([0, 3, 8, 12, 16]).apply_value({'theta': 0.529})
    expected = _cube_reference(circuit, values, n_qubits, repetitions=3)
    creator = MQVectorNPU(n_qubits)
    program = creator.compile(circuit)
    states = [creator.copy(), MQVectorNPU(n_qubits), creator.copy()]
    del creator
    for sim in states:
        sim.set_qs(values)

    def apply(sim):
        sim.apply_compiled(program, repetitions=3)
        return sim.get_qs()

    with ThreadPoolExecutor(max_workers=3) as pool:
        actual = list(pool.map(apply, states))
    for output in actual:
        assert np.linalg.norm(output - expected) < 2e-6


@pytest.mark.parametrize('n_qubits,seed', [(13, 19), (17, 1729), (20, 4099)])
def test_cube_generic_random_circuit(n_qubits, seed):
    circuit = random_circuit(n_qubits, n_qubits * 100, seed=seed)
    values = _state(n_qubits, seed=seed + 1)
    expected = _cube_reference(circuit, values, n_qubits)
    sim = MQVectorNPU(n_qubits)
    program = sim.compile(circuit)
    assert program.cube_tile_count > 0
    sim.set_qs(values)
    sim.apply_compiled(program)
    assert np.linalg.norm(sim.get_qs() - expected) < 3e-5


@pytest.mark.parametrize('n_qubits,width', [(12, 4), (13, 5), (14, 6), (15, 4), (22, 6)])
def test_cube_partial_batches_idle_workers_and_workspace_reuse(n_qubits, width):
    targets = [0, 2, 4, n_qubits - 1, n_qubits - 3, n_qubits - 5][:width]
    values = _state(n_qubits, seed=907)
    circuit = _cube_region(targets).apply_value({'theta': -0.387})
    expected = _cube_reference(circuit, values, n_qubits, repetitions=2)
    sim = MQVectorNPU(n_qubits)
    program = sim.compile(circuit)
    assert program.cube_tile_count == 1
    assert program.tile_count == 2
    sim.set_qs(values)
    sim.apply_compiled(program, repetitions=2)
    assert np.linalg.norm(sim.get_qs() - expected) < 2e-6


@pytest.mark.parametrize('minimum', [0, 4])
def test_custom_gate_capability_does_not_depend_on_fusion(minimum):
    sim = MQVectorNPU(13, cube_min_gates=minimum)
    sim.set_qs(_state(13, seed=307))
    before = sim.get_qs()
    circuit = _cube_region([0, 3, 8, 12]).apply_value({'theta': 0.913})
    circuit += G.UnivMathGate('unsupported', np.eye(128)).on([0, 1, 2, 3, 4, 8, 12])
    with pytest.raises(ValueError, match='six target'):
        sim.apply_circuit(circuit)
    np.testing.assert_array_equal(sim.get_qs(), before)


def test_native_and_python_defaults_select_the_same_cube_policy():
    from mindquantum import _mq_vector_npu  # pylint: disable=import-outside-toplevel
    from mindquantum.core.parameterresolver import ParameterResolver

    sim = MQVectorNPU(13)
    circuit = Circuit([G.U3(0.371 + i / 19, -0.827, 1.219).on(12) for i in range(4)])
    native = _mq_vector_npu.compile(circuit.get_cpp_obj(), ParameterResolver(), 13)
    assert sim.sim.prepare(native).cube_tile_count == sim.compile(circuit).cube_tile_count == 1


@pytest.mark.parametrize('n_qubits,width', [(12, 4), (13, 5), (14, 6), (15, 4), (18, 5), (22, 6)])
def test_cube_layout_restores_canonical_basis_before_replay(n_qubits, width):
    targets = [0, 2, 4, n_qubits - 1, n_qubits - 3, n_qubits - 5][:width]
    values = _state(n_qubits, seed=853)
    circuit = _cube_region(targets).apply_value({'theta': -0.381})
    expected = _cube_reference(circuit, values, n_qubits, repetitions=2)
    sim = MQVectorNPU(n_qubits)
    program = sim.compile(circuit)
    assert program.cube_tile_count == 1
    assert program.tile_count == 2
    sim.set_qs(values)
    sim.apply_compiled(program, repetitions=2)
    assert np.linalg.norm(sim.get_qs() - expected) < 2e-6
    sim.apply_compiled(sim.compile(circuit.hermitian()), repetitions=2)
    assert np.linalg.norm(sim.get_qs() - values) < 3e-6


@pytest.mark.parametrize('n_qubits', [13, 17, 21])
def test_cube_layout_changing_support_and_vector_predicates(n_qubits):
    values = _state(n_qubits, seed=913)
    circuit = _cube_region([0, 3, 7, n_qubits - 2]).apply_value({'theta': 0.513})
    circuit += G.RotPauliString('XYZZZZZ', -0.719).on([0, 2, 4, 6, 8, 10, 12])
    circuit += G.RotPauliString('IZZXYZI', 0.873).on([1, 4, 7, 8, 9, 11, 12], [0])
    circuit += G.U3(-0.39, 0.783, -0.217).on(1, [2, 4, 6, 8, 10, 12])
    circuit += G.RZ(-0.823).on(9, [1, 3, 5, 7, 11, 12])
    circuit += _cube_region([1, 4, 8, n_qubits - 1]).apply_value({'theta': -0.247})
    # Arbitrary nonunitary matrices expose basis-order and zero-row mistakes.
    matrix = np.array([[0, 0, 0, 0], [0.19j, 0.38, -0.73j, 0], [0, 0.42j, 0, 0.23], [0.41, 0, -0.32, 0.67j]])
    circuit += G.UnivMathGate('asymmetric-layout', matrix).on([n_qubits - 3, 0], [1, 3, 5, 7, 9])
    circuit += _cube_region([2, 5, 9, n_qubits - 2]).apply_value({'theta': -0.317})
    sim = MQVectorNPU(n_qubits)
    program = sim.compile(circuit)
    assert program.cube_tile_count >= 3
    assert program.tile_count > program.cube_tile_count
    sim.set_qs(values)
    sim.apply_compiled(program)
    # No intervening readback: restoration must happen during execution, not
    # lazily in get_qs. The next call must already see canonical qubit labels.
    suffix = Circuit([G.RY(0.913).on(0, n_qubits - 1), G.X.on(n_qubits - 2, 3)])
    sim.apply_circuit(suffix)
    # Re-setting a nonunitary intermediate state would normalize the reference.
    expected = _cube_reference(circuit + suffix, values, n_qubits)
    assert np.linalg.norm(sim.get_qs() - expected) < 3e-6


def test_cube_layout_avoids_identity_restoration():
    n_qubits = 18
    values = _state(n_qubits, seed=167)
    # These target bits already occupy the GEMM row positions in their packet.
    circuit = _cube_region([12, 13, 14, 15, 16, 17]).apply_value({'theta': -0.931})
    sim = MQVectorNPU(n_qubits)
    program = sim.compile(circuit)
    assert program.cube_tile_count == program.tile_count == 1
    sim.set_qs(values)
    sim.apply_compiled(program)
    assert np.linalg.norm(sim.get_qs() - _cube_reference(circuit, values, n_qubits)) < 2e-6


def test_cube_layout_restoration_uses_cube_not_vector_packet_budget():
    n_qubits = 14
    values = _state(n_qubits, seed=523)
    circuit = Circuit([G.U3(0.371 + i / 83, -0.827, 1.219).on(13) for i in range(41)])
    sim = MQVectorNPU(n_qubits, tile_qubits=4, fusion_qubits=1)
    program = sim.compile(circuit)
    assert program.cube_tile_count == 1
    assert program.tile_count == 2
    sim.set_qs(values)
    sim.apply_compiled(program, repetitions=3)
    expected = _cube_reference(circuit, values, n_qubits, repetitions=3)
    assert np.linalg.norm(sim.get_qs() - expected) < 2e-6
