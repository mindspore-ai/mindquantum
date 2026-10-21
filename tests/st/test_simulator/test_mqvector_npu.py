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

pytestmark = [
    pytest.mark.platform_arm_ascend910b_training,
    pytest.mark.skipif(find_spec('mindquantum._mq_vector_npu') is None, reason='Ascend extension not built'),
]


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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_reset_roundtrip_and_copy(n_qubits):
    """Feature: State ownership and padding.
    Description: Reset one state after copying arbitrary amplitudes at several sizes.
    Expectation: Reset yields zero basis and the copy retains every original amplitude.
    """
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_readback_snapshot_owns_storage_independent_of_device_and_views():
    """Feature: Readback lifetime.
    Description: Retain an array view after destroying the simulator and parent array.
    Expectation: The view still contains the original amplitudes.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_asymmetric_matrix_across_dma_and_external_controls(target, controls):
    """Feature: Controlled matrix evolution.
    Description: Move a complex asymmetric gate across local and external target bits.
    Expectation: The complete state agrees with independent little-endian embedding.
    """
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_strided_pair_spans_and_masked_tails():
    """Feature: Controlled matrices on aligned and fragmented amplitude groups.
    Description: Exercise repeat boundaries and small active sectors with many controls.
    Expectation: No active amplitude is lost and inactive amplitudes remain unchanged.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_builtin_gates_and_inverse(gate):
    """Feature: Built-in gate semantics.
    Description: Apply each supported gate and its inverse to a complex state.
    Expectation: Forward amplitudes match NumPy and inverse evolution restores the input.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_dense_two_qubit_matrix(targets):
    """Feature: Custom matrix target ordering.
    Description: Apply a dense two-qubit matrix at different ordered locations.
    Expectation: Amplitudes agree with NumPy without transposition or conjugation.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_fusion_matches_complex128_and_preserves_norm(tile_qubits, fusion_qubits):
    """Feature: Configurable tiling.
    Description: Run a mixed circuit and its inverse under different fusion budgets.
    Expectation: Every budget agrees with complex128 CPU evolution and preserves norm.
    """
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
    reference.apply_circuit(circuit)
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_compiled_snapshot_reuse_lifetime_and_threads(cube):
    """Feature: Immutable reusable executables.
    Description: Share one executable across independent states after deleting its creator.
    Expectation: Concurrent replay ignores later source edits and matches the CPU reference.
    """
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
        reference.apply_circuit(original)
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_mixed_cube_vector_ordering_and_explicit_vector_policy():
    """Feature: Cube and Vector composition.
    Description: Force wide-control Vector gates between local Cube regions.
    Expectation: Mixed and Vector-only policies produce the same complex128 reference state.
    """
    local = Circuit([G.U3(0.17 + i / 127, -0.37, 0.29).on(i % 6) for i in range(48)])
    circuit = local + Circuit([G.X.on(0, 13), G.RY(-0.73).on(13, 0)]) + local.hermitian()
    values = _state(16)
    reference = Simulator('mqvector', 16, dtype=mq.complex128)
    reference.set_qs(values)
    reference.apply_circuit(circuit)
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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_cube_rejection_before_state_mutation_and_selection_boundary():
    """Feature: Cube eligibility and fail-fast compilation.
    Description: Cross the fusion threshold and append invalid gates after valid ones.
    Expectation: Selection respects the policy and failed compilation leaves state untouched.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_caller_context_is_preserved_during_use_and_destruction(n_qubits):
    """Feature: Ascend context isolation.
    Description: Create, use and destroy backend objects inside a foreign runtime context.
    Expectation: Each operation restores the caller's context.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_general_random_circuit_n_times_100(n_qubits, seed):
    """Feature: Deep mixed-gate evolution.
    Description: Execute seeded circuits with one hundred gates per qubit.
    Expectation: Full-state error and norm drift stay within the complex64 tolerance.
    """
    circuit = random_circuit(n_qubits, n_qubits * 100, seed=seed)
    values = _state(n_qubits, seed=seed + 1)
    reference = Simulator('mqvector', n_qubits, dtype=mq.complex128)
    reference.set_qs(values)
    reference.apply_circuit(circuit)
    sim = MQVectorNPU(n_qubits)
    sim.set_qs(values)
    executable = sim.compile(circuit)
    assert 0 < executable.cube_tile_count <= executable.tile_count
    sim.apply_compiled(executable)
    actual = sim.get_qs().astype(np.complex128)
    assert np.linalg.norm(actual - reference.get_qs()) < 2e-5
    assert abs(np.vdot(actual, actual).real - 1) < 2e-5


@pytest.mark.parametrize('target', [0, 7])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_matrix_zero_row_and_delayed_first_nonzero_contribution(target):
    """Feature: Nonunitary custom matrix rows.
    Description: Use zero rows and imaginary first contributions at low and high targets.
    Expectation: No stale scratch values enter the result.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_complex_plane_dependencies_and_zero_rows_across_dense_operations(controls):
    """Feature: Consecutive complex matrix operations.
    Description: Alternate asymmetric matrices and target orders while reusing device scratch.
    Expectation: Both complex planes match the independently embedded matrix products.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_disjoint_matrix_blocks_share_indices_and_preserve_predicates(targets, controls, instructions):
    """Feature: Controlled Pauli evolution.
    Description: Combine external parity with disjoint active sectors and padded rows.
    Expectation: Each predicate selects exactly its intended amplitudes.
    """
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
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_compact_support_spectators_and_diagonal_predicates(tile_qubits, fusion_qubits):
    """Feature: Spectator and control distinction.
    Description: Move controls and diagonal operands across tile boundaries by changing width.
    Expectation: All widths preserve the same circuit semantics.
    """
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
    reference.apply_circuit(circuit)
    sim = MQVectorNPU(15, tile_qubits=tile_qubits, fusion_qubits=fusion_qubits)
    sim.set_qs(values)
    sim.apply_circuit(circuit)
    assert np.linalg.norm(sim.get_qs() - reference.get_qs()) < 2e-6


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_index_reuse_does_not_alias_coefficients_or_executable_storage():
    """Feature: Independent executable ownership.
    Description: Compile identical access patterns with different matrices and delete one executable.
    Expectation: Each retained executable uses its own coefficients and remains replayable.
    """
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
        reference.apply_circuit(circuit)
        assert np.linalg.norm(sim.get_qs() - reference.get_qs()) < 3e-6
    expected = sim.get_qs()
    del first, executable
    gc.collect()
    sim.set_qs(values)
    sim.apply_compiled(second)
    np.testing.assert_array_equal(sim.get_qs(), expected)


@pytest.mark.parametrize('bad_gate', [G.Measure(), G.DepolarizingChannel(0.2)])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_stochastic_circuit_cannot_be_frozen_as_deterministic_executable(bad_gate):
    """Feature: Deterministic executable boundary.
    Description: Try compiling measurement and noise gates as reusable deterministic programs.
    Expectation: Compilation fails without changing the state.
    """
    sim = MQVectorNPU(4)
    sim.set_qs(_state(4))
    before = sim.get_qs()
    circuit = Circuit([G.H.on(0), bad_gate.on(1)])
    with pytest.raises(ValueError, match='without measurement or noise'):
        sim.compile(circuit)
    np.testing.assert_array_equal(sim.get_qs(), before)


def _cube_region(targets):
    circuit = Circuit()
    for i in range(41):
        target = targets[i % len(targets)]
        control = targets[(i + 1) % len(targets)]
        circuit += G.U3(0.371 + i / 83, -0.827, 1.219).on(target, control if i % 3 and len(targets) > 1 else [])
    if len(targets) > 1:
        circuit += G.FSim(0.712, -0.913).on([targets[-1], targets[0]])
        circuit += G.Rzz(-0.497).on([targets[-2], targets[1]])
    circuit += G.GlobalPhase(0.139).on(targets[0])
    circuit += G.RX('theta').on(targets[-1])
    return circuit


def _cube_reference(circuit, values, n_qubits, repetitions=1):
    reference = Simulator('mqvector', n_qubits, dtype=mq.complex128)
    reference.set_qs(values)
    for _ in range(repetitions):
        reference.apply_circuit(circuit)
    return reference.get_qs()


@pytest.mark.parametrize(
    'n_qubits,targets,tile,fusion',
    [
        (12, [0, 2, 4, 11], 12, 6),
        (13, [0, 2, 4, 12, 10], 12, 6),
        (14, [0, 2, 4, 13, 11, 9], 12, 6),
        (15, [0, 2, 4, 14], 12, 6),
        (18, [1, 4, 9, 14, 17], 12, 6),
        (18, [12, 13, 14, 15, 16, 17], 12, 6),
        (22, [0, 2, 4, 21, 19, 17], 12, 6),
        (14, [13], 4, 1),
    ],
)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_cube_replay_preserves_canonical_layout_across_packet_sizes(n_qubits, targets, tile, fusion):
    """Feature: Public basis order after Cube execution.
    Description: Replay noncontiguous, already-packed and narrow-budget circuits across batch boundaries.
    Expectation: Repeated forward and inverse calls match canonical CPU evolution.
    """
    values = _state(n_qubits, seed=592)
    circuit = _cube_region(targets)
    bound = circuit.apply_value({'theta': -0.674})
    expected = _cube_reference(bound, values, n_qubits, repetitions=2)
    sim = MQVectorNPU(n_qubits, tile_qubits=tile, fusion_qubits=fusion)
    sim.set_qs(values)
    program = sim.compile(circuit, {'theta': -0.674})
    assert program.cube_tile_count == 1
    sim.apply_compiled(program, repetitions=2)
    assert np.linalg.norm(sim.get_qs() - expected) < 2e-6
    sim.apply_compiled(sim.compile(bound.hermitian()), repetitions=2)
    assert np.linalg.norm(sim.get_qs() - values) < 2e-6


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_cube_control_dependencies_and_wide_pauli_boundaries():
    """Feature: Wide support at fusion boundaries.
    Description: Change external controls between dense regions and include a seven-operand Pauli word.
    Expectation: Fusion never drops a control or diagonal operand.
    """
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


@pytest.mark.parametrize('minimum', [0, 4])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_custom_gate_capability_does_not_depend_on_fusion(minimum):
    """Feature: Policy-independent capabilities.
    Description: Append an unsupported custom gate with Cube enabled and disabled.
    Expectation: Both policies reject the complete circuit before state mutation.
    """
    sim = MQVectorNPU(13, cube_min_gates=minimum)
    sim.set_qs(_state(13, seed=307))
    before = sim.get_qs()
    circuit = _cube_region([0, 3, 8, 12]).apply_value({'theta': 0.913})
    circuit += G.UnivMathGate('unsupported', np.eye(128)).on([0, 1, 2, 3, 4, 8, 12])
    with pytest.raises(ValueError, match='six target'):
        sim.apply_circuit(circuit)
    np.testing.assert_array_equal(sim.get_qs(), before)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_native_and_python_defaults_select_the_same_cube_policy():
    """Feature: Default execution policy.
    Description: Compile one circuit through native and Python entry points.
    Expectation: Both select Cube at the same threshold.
    """
    from mindquantum import _mq_vector_npu  # pylint: disable=import-outside-toplevel
    from mindquantum.core.parameterresolver import ParameterResolver

    sim = MQVectorNPU(13)
    circuit = Circuit([G.U3(0.371 + i / 19, -0.827, 1.219).on(12) for i in range(4)])
    native = _mq_vector_npu.compile(circuit.get_cpp_obj(), ParameterResolver(), 13)
    assert sim.sim.prepare(native).cube_tile_count == sim.compile(circuit).cube_tile_count == 1


@pytest.mark.parametrize('n_qubits', [13, 17, 21])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_cube_layout_changing_support_and_vector_predicates(n_qubits):
    """Feature: Layout propagation through mixed circuits.
    Description: Change dense support around wide predicates and apply a suffix without readback.
    Expectation: The suffix sees canonical qubit labels, even for a nonunitary intermediate state.
    """
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
