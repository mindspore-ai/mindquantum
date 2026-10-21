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
"""Public mqvector_npu contracts, checked against complex128 CPU/NumPy references."""

import gc
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import mindquantum as mq
from mindquantum.core import gates as G  # noqa: N812
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator import Simulator, get_supported_simulator, inner_product
from mindquantum.simulator.available_simulator import MQVECTOR_NPU_SUPPORTED
from mindquantum.simulator.mqvector_npu import MQVectorNPU

pytestmark = [
    pytest.mark.platform_arm_ascend910b_training,
    pytest.mark.skipif(not MQVECTOR_NPU_SUPPORTED, reason='Ascend extension not built'),
]


def _state(n, seed=17):
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return vector / np.linalg.norm(vector)


def _sim(n, state=None, seed=19):
    sim = Simulator('mqvector_npu', n, seed=seed, device_id=0)
    if state is not None:
        sim.set_qs(state)
    return sim


def _ham():
    return Hamiltonian(QubitOperator('Z0', 0.7) + QubitOperator('Y1', -0.4) + QubitOperator('X0 X1', 0.2), mq.float32)


def _cpu_state(state, circuit, pr=None):
    sim = Simulator('mqvector', int(np.log2(len(state))), dtype=mq.complex128)
    sim.set_qs(state)
    sim.apply_circuit(circuit, pr)
    return sim.get_qs()


def _no_download(*_args, **_kwargs):
    raise AssertionError('Operation must not download the state vector.')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_import_registration_does_not_load_npu():
    """Feature: Lazy backend registration.
    Description: Import MindQuantum and inspect supported backends in a fresh interpreter.
    Expectation: Registration does not import the native NPU extension.
    """
    subprocess.run(
        [
            sys.executable,
            '-c',
            "import sys; import mindquantum; from mindquantum.simulator import get_supported_simulator; "
            "from mindquantum.simulator.mqvector_npu import MQVectorNPU; "
            "assert 'mindquantum._mq_vector_npu' not in sys.modules; "
            "assert ['mqvector_npu', mindquantum.complex64] in get_supported_simulator()",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_constructor_precision_and_errors_do_not_allocate_or_retry(monkeypatch):
    """Feature: Fail-fast construction.
    Description: Supply unsupported precision and policies before an unavailable device.
    Expectation: Invalid inputs allocate nothing and a device error propagates without retry.
    """
    from mindquantum import _mq_vector_npu as native

    calls = []

    def unavailable(*args):
        calls.append(args)
        raise RuntimeError('device unavailable')

    monkeypatch.setattr(native, 'State', unavailable)
    for dtype in (mq.complex128, mq.float32, np.complex64):
        with pytest.raises(TypeError, match='complex64 only'):
            Simulator('mqvector_npu', 1, dtype=dtype)
    for qubits, tile, fusion in ((33, 12, 6), (8, 3, 1), (8, 4, 2), (8, 13, 6), (8, 7, 5)):
        with pytest.raises(ValueError):
            MQVectorNPU(qubits, tile_qubits=tile, fusion_qubits=fusion)
        with pytest.raises(ValueError):
            native.ForwardPlan([], qubits, tile, fusion)
        with pytest.raises(ValueError):
            native.AdjointPlan([], {}, qubits, tile, fusion)
    assert not calls
    with pytest.raises(RuntimeError, match='device unavailable'):
        Simulator('mqvector_npu', 1)
    assert len(calls) == 1
    assert ['mqvector_npu', mq.complex64] in get_supported_simulator()
    assert ['mqvector_npu', mq.complex128] not in get_supported_simulator()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_state_copy_reset_astype_and_direct_backend():
    """Feature: Simulator state API.
    Description: Normalize large inputs, copy and convert state, then reject malformed inputs.
    Expectation: Valid operations preserve independent ownership and errors preserve the old state.
    """
    state = _state(2)
    sim = Simulator(MQVectorNPU(2, seed=5))
    sim.set_qs(state * 1e200)
    sim.apply_circuit(Circuit())
    assert sim.dtype == mq.complex64
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)
    other = sim.copy()
    other.apply_gate(G.X.on(0))
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)
    converted = sim.astype(mq.complex64, seed=32)
    np.testing.assert_array_equal(converted.get_qs(), sim.get_qs())
    with pytest.raises(TypeError, match='complex64 only'):
        sim.astype(mq.complex128)
    assert isinstance(sim.get_qs(ket=True), str)
    for invalid in ([0, 0, 0, 0], [1, np.nan, 0, 0], [[1, 0], [0, 0]], [1, 0]):
        with pytest.raises(ValueError):
            sim.set_qs(invalid)
        np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)
    sim.reset()
    np.testing.assert_array_equal(sim.get_qs(), [1, 0, 0, 0])
    assert sim.entropy() == 0
    assert sim.purity() == 1
    np.testing.assert_array_equal(sim.get_pure_state_vector(), sim.get_qs())


@pytest.mark.parametrize('n,controls', [(3, [2]), (15, [1, 12, 14])])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_evolution_derivative_and_circuit_matrix(n, controls):
    """Feature: Evolution, intrinsic derivatives and circuit matrices.
    Description: Differentiate a no_grad gate with local and multiple external controls.
    Expectation: Active amplitudes equal dU times the input and every inactive sector is zero.
    """
    state = _state(n)
    circuit = Circuit([G.RX('a').on(2, 0), G.U3('b', 0.2, 'a').on(0), G.Rxy(0.6).on([1, 2]), G.S.on(1)])
    pr = {'a': 0.4, 'b': -0.8}
    expected = _cpu_state(state, circuit, pr)
    sim = _sim(n, state)
    sim.apply_circuit(circuit, pr)
    np.testing.assert_allclose(sim.get_qs(), expected, atol=5e-7)
    gate = G.RY(ParameterResolver('a').no_grad()).on(0, controls)
    sim.set_qs(state)
    sim.apply_gate(gate, {'a': 0.3}, diff=True)
    indices = np.arange(1 << n)
    control_mask = sum(1 << q for q in controls)
    bases = indices[((indices & 1) == 0) & ((indices & control_mask) == control_mask)]
    pairs = np.stack([bases, bases | 1])
    expected = np.zeros_like(state)
    expected[pairs] = G.RY('a').diff_matrix({'a': 0.3}) @ state[pairs]
    np.testing.assert_allclose(sim.get_qs(), expected, atol=3e-7)
    matrix_circuit = Circuit().ry('a', 0).s(0).rx(0.6, 1).x(1, 0)
    np.testing.assert_allclose(
        matrix_circuit.matrix({'a': 0.2}, backend='mqvector_npu'),
        matrix_circuit.matrix({'a': 0.2}, dtype=mq.complex128),
        atol=4e-7,
    )


@pytest.mark.parametrize('stochastic', [False, True])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_late_unsupported_gate_and_parameter_failures_are_atomic(stochastic):
    """Feature: Atomic input rejection.
    Description: Put unsupported gates or invalid parameters after valid and stochastic operations.
    Expectation: Neither state nor RNG advances on rejection.
    """
    state = _state(7)
    sim = _sim(7, state)
    circuit = Circuit().h(0)
    if stochastic:
        circuit += G.Measure('first').on(0)
        circuit += G.BitFlipChannel(0.4).on(1)
    circuit += G.UnivMathGate('unsupported_dense', np.eye(128)).on(list(range(7)))
    control = sim.copy()
    with pytest.raises(ValueError, match='at most six'):
        sim.apply_circuit(circuit)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)
    for pr in ({}, {'a': np.nan}, {'a': np.inf}, {'a': 1j}, {'a': 1.0, 'unused': np.inf}):
        with pytest.raises(ValueError):
            sim.apply_circuit(Circuit().x(0).ry('a', 1), pr)
        np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)
    probe = Circuit().h(0).measure('coin', 0)
    np.testing.assert_array_equal(sim.sampling(probe, shots=64).samples, control.sampling(probe, shots=64).samples)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_inner_product_hamiltonian_and_asymmetric_expectation(monkeypatch):
    """Feature: Complex observables with independent bra and ket.
    Description: Apply a nonsymmetric complex CSR operator to different circuit-evolved states.
    Expectation: Device reductions match NumPy and expectation leaves both source states unchanged.
    """
    from mindquantum import _mq_vector_npu as native

    right_state, left_state = _state(2, 11), _state(2, 12)
    right, left = _sim(2, right_state), _sim(2, left_state)
    raw = np.array([[0.3j, 0.2, 0, -0.4j], [0, -0.2, 0.3j, 0], [0.7, 0, 0.1, 0], [0, 0.6j, 0.2, -0.5]])
    ham = Hamiltonian(csr_matrix(raw), mq.complex64)
    circ_right, circ_left = Circuit().rx('a', 0).ry(0.2, 1), Circuit().rz('a', 1).h(0)
    pr = {'a': 0.8}
    expected = np.vdot(_cpu_state(left_state, circ_left, pr), raw @ _cpu_state(right_state, circ_right, pr))
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        np.testing.assert_allclose(inner_product(left, right), np.vdot(left_state, right_state), atol=4e-7)
        np.testing.assert_allclose(right.get_expectation(ham, circ_right, circ_left, left, pr), expected, atol=5e-7)
    np.testing.assert_allclose(right.get_qs(), right_state, atol=2e-7)
    np.testing.assert_allclose(left.get_qs(), left_state, atol=2e-7)
    right.apply_hamiltonian(ham)
    np.testing.assert_allclose(right.get_qs(), raw @ right_state, atol=3e-7)
    with pytest.raises(TypeError, match='single-precision'):
        right.get_expectation(ham.astype(mq.complex128))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_terminal_sampling_reset_repetition_and_seed(monkeypatch):
    """Feature: Terminal measurement semantics.
    Description: Sample entangled qubits with reset and repeated measurement names.
    Expectation: Outcomes, ordering and seeded replay agree without downloading amplitudes.
    """
    from mindquantum import _mq_vector_npu as native

    sim = _sim(2)
    circuit = Circuit().h(0).x(1, 0)
    circuit += G.Measure('first', reset_to=0).on(0)
    circuit += G.Measure('repeat').on(0)
    circuit += G.Measure('other').on(1)
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        result = sim.sampling(circuit, shots=800, seed=3)
    assert result.keys == ['other', 'repeat', 'first']
    assert set(result.data) == {'000', '101'}
    assert abs(result.data['101'] / 800 - 0.5) < 0.07
    np.testing.assert_array_equal(result.samples, sim.sampling(circuit, shots=800, seed=3).samples)
    np.testing.assert_array_equal(sim.get_qs(), [1, 0, 0, 0])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_measurement_reset_midcircuit_and_independent_rng_copy():
    """Feature: Midcircuit trajectories and copied RNG state.
    Description: Evolve after reset measurements and draw from two copied simulators.
    Expectation: Resets affect later gates and copied RNGs advance independently but reproducibly.
    """
    sim = _sim(2)
    circuit = Circuit().h(0).x(1, 0)
    circuit += G.Measure('first', reset_to=1).on(0)
    circuit.x(1, 0)
    circuit += G.Measure('last').on(1)
    result = sim.sampling(circuit, shots=200, seed=7)
    assert set(result.data) == {'01', '10'}
    applied = sim.apply_circuit(circuit)
    assert applied.shots == 1
    assert applied.samples[0].sum() == 1
    assert abs(sim.get_qs()[1]) + abs(sim.get_qs()[3]) == pytest.approx(1, abs=3e-7)
    sim.reset()
    sim.apply_gate(G.X.on(0))
    assert sim.apply_gate(G.Measure(reset_to=0).on(0)) == 1
    np.testing.assert_array_equal(sim.get_qs(), [1, 0, 0, 0])
    other = sim.copy()
    sequence = Circuit().h(0).measure('coin', 0)
    a = [sim.apply_circuit(sequence).samples[0][0] for _ in range(16)]
    b = [other.apply_circuit(sequence).samples[0][0] for _ in range(16)]
    assert a == b


@pytest.mark.parametrize(
    'channel',
    [
        G.PauliChannel(0.13, 0.17, 0.19),
        G.BitFlipChannel(0.35),
        G.PhaseFlipChannel(0.4),
        G.BitPhaseFlipChannel(0.2),
        G.DepolarizingChannel(4 / 3),
        G.AmplitudeDampingChannel(0.37),
        G.PhaseDampingChannel(0.6),
        G.KrausChannel('damping', G.AmplitudeDampingChannel(0.37).matrix()),
        G.ThermalRelaxationChannel(100, 70, 35),
        G.ThermalRelaxationChannel(100, 150, 35),
    ],
)
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_noise_trajectories_match_small_density_matrix(channel, monkeypatch):
    """Feature: Supported noise channels.
    Description: Sample a noisy rotated state using seeded trajectories.
    Expectation: Frequencies agree with a density-matrix reference without amplitude downloads.
    """
    from mindquantum import _mq_vector_npu as native

    state = _state(1, 31)
    rho = np.outer(state, state.conj())
    expected = sum(matrix @ rho @ matrix.conj().T for matrix in channel.matrix())
    rotation = G.RY(0.7).matrix()
    expected = rotation @ expected @ rotation.conj().T
    circuit = Circuit([channel.on(0)]).ry(0.7, 0).measure('result', 0)
    sim = _sim(1, state)
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        result = sim.sampling(circuit, shots=512, seed=42)
    assert result.data.get('1', 0) / 512 == pytest.approx(expected[1, 1].real, abs=0.08)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_grouped_and_multiqubit_depolarizing_do_not_expand_kraus(monkeypatch):
    """Feature: Wide Pauli noise.
    Description: Apply grouped and multiqubit depolarizing channels without exponential matrix expansion.
    Expectation: States remain normalized and sampled probabilities match the analytic channel.
    """
    monkeypatch.setattr(G.GroupedPauliChannel, 'matrix', _no_download)
    monkeypatch.setattr(G.DepolarizingChannel, 'matrix', _no_download)
    sim = _sim(8)
    grouped = G.GroupedPauliChannel(np.tile([1.0, 0.0, 0.0], (8, 1))).on(list(range(8)))
    sim.apply_gate(grouped)
    expected = np.zeros(256)
    expected[-1] = 1
    np.testing.assert_array_equal(sim.get_qs(), expected)
    sim.apply_gate(G.DepolarizingChannel(0.3, 8).on(list(range(8))))
    assert np.linalg.norm(sim.get_qs()) == pytest.approx(1, abs=3e-7)
    small = _sim(2)
    circuit = Circuit([G.DepolarizingChannel(16 / 15, 2).on([0, 1])]).measure_all()
    result = small.sampling(circuit, shots=512, seed=8)
    for bitstring in ('00', '01', '10', '11'):
        probability = (3 if bitstring == '00' else 4) / 15
        assert result.data.get(bitstring, 0) / 512 == pytest.approx(probability, abs=0.075)


@pytest.mark.parametrize('side', ['implicit', 'identical', 'distinct'])
@pytest.mark.parametrize('csr', [False, True])
@pytest.mark.parametrize('pr_shift', [False, True])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_gradient_snapshots_ignore_later_source_and_metadata_mutations(side, csr, pr_shift):
    """Feature: Gradient snapshot isolation.
    Description: Mutate source circuits, observables and public metadata after creating an evaluator.
    Expectation: Adjoint and shift evaluation retain their original semantics on either side.
    """
    sim = _sim(2)
    right = Circuit().h(0).x(1, 0).ry('a', 0)
    left = Circuit().rx('a', 1).h(0) if side == 'distinct' else right
    kwargs = {} if side == 'implicit' else {'circ_left': left}
    ham = Hamiltonian(csr_matrix(np.diag([1, 0.3j, -0.5, 0.2j])), mq.complex64) if csr else _ham()
    hams = [ham]
    operator = sim.get_expectation_with_grad(hams, right, pr_shift=pr_shift, **kwargs)
    first = operator([0.3])
    right.ry('later_right', 0)
    left.rz('later_left', 1)
    if csr:
        ham.sparse_matrix.data[:] = 0
    else:
        ham.ham_termlist.clear()
    hams.clear()
    operator.ansatz_params_name[:] = ['mutable', 'metadata']
    for got, want in zip(operator([0.3]), first):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('pr_shift', [False, True])
@pytest.mark.parametrize('representation', ['pauli', 'sparse_pauli', 'csr'])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_gradient_snapshots_current_hamiltonian_data_not_old_native_caches(pr_shift, representation):
    """Feature: Observable snapshot freshness.
    Description: Populate native Hamiltonian caches, edit the source and construct a gradient.
    Expectation: The gradient captures current coefficients, not stale caches or subsequent edits.
    """
    if representation == 'csr':
        ham = Hamiltonian(csr_matrix(np.diag([0.7 + 0.4j, -0.3 + 0.8j])), mq.complex64)
    else:
        ham = Hamiltonian(QubitOperator('Z0', 0.7), mq.float32)
        if representation == 'sparse_pauli':
            ham.sparse(1)
    # Warm both caches, then edit the source BEFORE creating the new plan.
    # A plan must snapshot the edited data, not those stale native objects.
    ham.get_cpp_obj()
    ham.get_cpp_obj(hermitian=True)
    if representation == 'csr':
        ham.sparse_matrix.data[:] = [0.2 - 0.9j, 0.6 + 0.3j]
        diagonal = np.asarray([0.2 - 0.9j, 0.6 + 0.3j])
    else:
        ham.ham_termlist[:] = [(((0, 'Z'),), -0.4)]
        diagonal = np.asarray([-0.4, 0.4])
    operator = _sim(1).get_expectation_with_grad(ham, Circuit().ry('a', 0), pr_shift=pr_shift)
    # Subsequent edits must not change the already-owned native snapshot.
    if representation == 'csr':
        ham.sparse_matrix.data[:] = 0
    else:
        ham.ham_termlist.clear()
    angle = 0.37
    value, gradient = operator([angle])
    expected_value = diagonal[0] * np.cos(angle / 2) ** 2 + diagonal[1] * np.sin(angle / 2) ** 2
    expected_gradient = (diagonal[1] - diagonal[0]) * np.sin(angle) / 2
    np.testing.assert_allclose(value, [[expected_value]], atol=3e-7)
    np.testing.assert_allclose(gradient, [[[expected_gradient]]], atol=3e-7)


@pytest.mark.parametrize('pr_shift', [False, True])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_batched_shared_parameters_no_grad_and_snapshots(pr_shift, monkeypatch):
    """Feature: Batched encoder and ansatz gradients.
    Description: Combine shared affine expressions and no_grad coefficients, then mutate wrapper metadata.
    Expectation: Output shapes and Jacobian-weighted values match complex128 for every sample.
    """
    from mindquantum import _mq_vector_npu as native

    state = _state(2, 36)
    sim = _sim(2, state)
    encoder = Circuit(
        [G.RX(ParameterResolver({'e': 0.6}).as_encoder()).on(0), G.RY(ParameterResolver('e').as_encoder()).on(1)]
    )
    circuit = encoder + Circuit(
        [
            G.RY(ParameterResolver({'a': 1.3, 'b': 0.2}, const=0.1).no_grad_part('b')).on(0, 1),
            G.RZ('a').on(1),
            G.RX('b').on(1),
        ]
    )
    ham = _ham()
    cpu = Simulator('mqvector', 2, dtype=mq.complex128)
    cpu.set_qs(state)
    reference = cpu.get_expectation_with_grad(ham.astype(mq.float64), circuit)
    operator = sim.get_expectation_with_grad(ham, circuit, parallel_worker=2, pr_shift=pr_shift)
    data, weights = np.array([[0.2], [-0.4], [0.7]]), np.array([0.3, -0.8])
    expected = reference(data, weights)
    circuit[2].coeff.const = 7
    ham.ham_termlist.clear()
    assert operator.circ_right is circuit
    assert operator.hams[0] is ham
    operator.circ_right.x(0).ry('later', 0)
    operator.hams[0].ham_termlist.clear()
    operator.hams.clear()
    operator.encoder_params_name.clear()
    operator.ansatz_params_name[:] = ['mutable', 'metadata', 'only']
    operator.circ_right = operator.circ_left = None
    operator.sim = None
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        actual = operator(data, weights)
    assert [value.shape for value in actual] == [(3, 1), (3, 1, 1), (3, 1, 2)]
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=3e-6)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)


@pytest.mark.parametrize('pr_shift', [False, True])
@pytest.mark.parametrize('different_sides', [False, True])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_nonhermitian_gradients_need_both_sweeps(pr_shift, different_sides):
    """Feature: Non-Hermitian gradients.
    Description: Use complex CSR with duplicate columns and optionally distinct circuits and initial states.
    Expectation: Both real and imaginary derivatives agree with independent finite differences.
    """
    state, left_state = _state(2, 45), _state(2, 46)
    sim, left_sim = _sim(2, state), _sim(2, left_state)
    raw = np.array([[0.3j, 0.2, 0, -0.4j], [0, -0.2, 0.3j, 0], [0.7, 0, 0.1, 0], [0, 0.6j, 0.2, -0.5]])
    sparse = csr_matrix(raw)
    # Duplicate columns are legal CSR entries; H-dagger must preserve their sum.
    sparse = csr_matrix((np.repeat(sparse.data / 2, 2), np.repeat(sparse.indices, 2), sparse.indptr * 2))
    ham = Hamiltonian(sparse, mq.complex64)
    right = Circuit([G.U3('a', {'b': 0.4}, 'a').on(0, 1), G.RY({'a': -0.3}).on(1)])
    left = Circuit().rx('b', 0).rz({'a': 0.7}, 1) if different_sides else right
    kwargs = {'circ_left': left, 'simulator_left': left_sim} if different_sides else {}
    operator = sim.get_expectation_with_grad(ham, right, pr_shift=pr_shift, **kwargs)
    pr = {'a': 0.32, 'b': -0.61}

    def expectation(parameters):
        ket = _cpu_state(state, right, parameters)
        bra = _cpu_state(left_state if different_sides else state, left, parameters)
        return np.vdot(bra, raw @ ket)

    f, grad = operator([pr[name] for name in operator.ansatz_params_name])
    np.testing.assert_allclose(f[0, 0], expectation(pr), atol=2e-6)
    for i, name in enumerate(operator.ansatz_params_name):
        delta = 1e-5
        plus, minus = dict(pr), dict(pr)
        plus[name] += delta
        minus[name] -= delta
        expected = (expectation(plus) - expectation(minus)) / (2 * delta)
        np.testing.assert_allclose(grad[0, 0, i], expected, atol=3e-6)


@pytest.mark.parametrize('gate', [G.Givens('a'), G.FSim('a', {'a': -0.3}), G.SWAPalpha('a')])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_exact_shift_for_multifrequency_gates(gate):
    """Feature: Exact multifrequency shift rules.
    Description: Differentiate controlled exchange gates sharing affine parameters with other rotations.
    Expectation: Shift results match CPU adjoints without finite-difference approximation.
    """
    state = _state(3, 51)
    circuit = Circuit([gate.on([0, 1], 2), G.RX({'a': 0.7}).on(2), G.GlobalPhase('a').on(1, 0)])
    ham = _ham()
    sim = _sim(3, state)
    cpu = Simulator('mqvector', 3, dtype=mq.complex128)
    cpu.set_qs(state)
    want = cpu.get_expectation_with_grad(ham.astype(mq.float64), circuit)([0.47])
    actual = sim.get_expectation_with_grad(ham, circuit, pr_shift=True)([0.47])
    for got, expected in zip(actual, want):
        np.testing.assert_allclose(got, expected, atol=4e-6)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_noisy_gradient_uses_exact_shift_and_shapes():
    """Feature: Noisy gradient snapshots and empty parameter sets.
    Description: Mutate a deterministic noise circuit after construction and evaluate encoder-only and empty circuits.
    Expectation: The snapshot follows its analytic derivative and all returned shapes remain valid.
    """
    sim = _sim(1)
    ham = Hamiltonian(QubitOperator('Z0'), mq.float32)
    # A deterministic channel still forces the trajectory/shift path, with an exact analytic answer.
    circuit = Circuit().ry({'a': 2}, 0) + G.BitFlipChannel(1).on(0)
    operator = sim.get_expectation_with_grad(ham, circuit)
    circuit[0].coeff['a'] = 7
    circuit[1].px = 0
    operator.circ_right = operator.circ_left = None
    f, grad = operator([0.2])
    np.testing.assert_allclose(f, [[-np.cos(0.4)]], atol=4e-7)
    np.testing.assert_allclose(grad, [[[2 * np.sin(0.4)]]], atol=8e-7)
    encoder = Circuit([G.RY(ParameterResolver('e').as_encoder()).on(0)])
    f, grad = sim.get_expectation_with_grad(ham, encoder)([[0.1], [0.4]])
    assert f.shape == (2, 1) and grad.shape == (2, 1, 1)
    f, grad = sim.get_expectation_with_grad(ham, Circuit())([])
    assert f.shape == (1, 1) and grad.shape == (1, 1, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_gradient_validation_and_nonunitary_gate():
    """Feature: Gradient capability boundary.
    Description: Supply invalid inputs, unsupported shift rules and a nonunitary custom gate.
    Expectation: Gradients reject them while ordinary nonunitary evolution remains supported.
    """
    sim = _sim(2)
    ham = _ham()
    for workers in (0, -1):
        with pytest.raises(ValueError, match='positive'):
            sim.get_expectation_with_grad(ham, Circuit().ry('a', 0), parallel_worker=workers)
    for circuit in (Circuit().measure('m', 0), Circuit().ry('a', 2)):
        with pytest.raises(ValueError):
            sim.get_expectation_with_grad(ham, circuit)
    operator = sim.get_expectation_with_grad(ham, Circuit().ry('a', 0))
    for values in ([np.nan], [np.inf], [1j], [[0.1]], [0.1, 0.2]):
        with pytest.raises(ValueError):
            operator(values)
    with pytest.raises(NotImplementedError, match='Exact parameter-shift.*Rn'):
        sim.get_expectation_with_grad(ham, Circuit([G.Rn('a', 0.2, 0.1).on(0)]), pr_shift=True)
    matrix = np.diag([1, 0.5]).astype(complex)
    circuit = Circuit().ry('a', 0) + G.UnivMathGate('nonunitary', matrix).on(0)
    with pytest.raises(ValueError, match='unitary'):
        sim.get_expectation_with_grad(ham, circuit)
    np.testing.assert_array_equal(sim.get_qs(), [1, 0, 0, 0])
    sim.apply_circuit(circuit, {'a': 0.3})
    np.testing.assert_allclose(sim.get_qs(), [np.cos(0.15), 0.5 * np.sin(0.15), 0, 0], atol=3e-7)


@pytest.mark.parametrize('cube', [0, 4])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_custom_derivatives_reject_nonunitary_bindings_without_mutation(cube):
    """Feature: Parameterized custom derivatives.
    Description: Bind a controlled callback gate at unitary and nonunitary parameter values.
    Expectation: Valid gradients match CPU and invalid bindings preserve the input state under both policies.
    """
    pytest.importorskip('numba')

    def matrix(angle):
        scale = 1 if angle < 0.5 else 0.5
        return np.array([[1 + 0j, 0j], [0j, scale * np.exp(1j * angle)]])

    def derivative(angle):
        scale = 1 if angle < 0.5 else 0.5
        return np.array([[0j, 0j], [0j, 1j * scale * np.exp(1j * angle)]])

    custom = G.gene_univ_parameterized_gate('staged_unitarity', matrix, derivative)
    state = _state(12, 723)
    sim = Simulator(MQVectorNPU(12, cube_min_gates=cube))
    sim.set_qs(state)
    cpu = Simulator('mqvector', 12, dtype=mq.complex128)
    cpu.set_qs(state)
    circuit = Circuit([G.RY('b').on(11), custom('a').on(11, 1)]) * 5
    ham = Hamiltonian(QubitOperator('Y11', 0.7) + QubitOperator('Z0', -0.2), mq.float32)
    op = sim.get_expectation_with_grad(ham, circuit)
    reference = cpu.get_expectation_with_grad(ham.astype(mq.float64), circuit)
    values = [{'a': 0.3, 'b': -0.41}[name] for name in op.ansatz_params_name]
    for got, want in zip(op(values), reference(values)):
        np.testing.assert_allclose(got, want, atol=2e-6)
    values[op.ansatz_params_name.index('a')] = 0.7
    with pytest.raises(ValueError, match='unitary'):
        op(values)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)


@pytest.mark.parametrize('explicit_left', [False, True])
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_custom_gradient_callbacks_outlive_mutated_and_deleted_sources(explicit_left):
    """Feature: Custom callback ownership.
    Description: Modify numeric data and callback attributes, then discard all public circuit references.
    Expectation: The evaluator retains its original callback and analytic phase gradient.
    """
    pytest.importorskip('numba')

    def matrix(angle):
        scale = 1 if angle < 0.5 else 0.5
        return np.array([[1 + 0j, 0j], [0j, scale * np.exp(1j * angle)]])

    def derivative(angle):
        scale = 1 if angle < 0.5 else 0.5
        return np.array([[0j, 0j], [0j, 1j * scale * np.exp(1j * angle)]])

    custom = G.gene_univ_parameterized_gate('private_callbacks', matrix, derivative)
    circuit = Circuit().h(0)
    circuit += G.UnivMathGate('fixed', np.diag([1, 1j])).on(0)
    circuit += custom(ParameterResolver({'a': 0.5}, const=0.1)).on(0)
    circuit += custom(0.1).on(0)
    sim = _sim(1)
    kwargs = {'circ_left': circuit} if explicit_left else {}
    operator = sim.get_expectation_with_grad(Hamiltonian(QubitOperator('X0'), mq.float32), circuit, **kwargs)
    circuit[1].matrix_value[:] = 0
    circuit[2].coeff['a'] = 7
    circuit[2].coeff.const = 0.8
    circuit[2].matrix_generator = lambda _: np.zeros((2, 2), dtype=complex)
    circuit[3].coeff.const = 0.9
    operator.circ_right = operator.circ_left = None
    operator.hams.clear()
    operator.ansatz_params_name.clear()
    kwargs.clear()
    del circuit
    gc.collect()
    for _ in range(2):
        value, gradient = operator([0.3])
        np.testing.assert_allclose(value, [[-np.sin(0.35)]], atol=3e-7)
        np.testing.assert_allclose(gradient, [[[-0.5 * np.cos(0.35)]]], atol=3e-7)
    with pytest.raises(ValueError, match='unitary'):
        operator([0.9])
    np.testing.assert_array_equal(sim.get_qs(), [1, 0])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_builtin_subclass_is_validated_and_its_numeric_data_is_snapshotted():
    """Feature: Custom subclasses are not trusted built-ins.
    Description: Mutate a rotation subclass's matrix scale after constructing a valid gradient.
    Expectation: The old snapshot remains valid and a new nonunitary binding is rejected.
    """

    class CustomRY(G.RY):
        def __init__(self, pr):
            super().__init__(pr)
            self.scale = np.ones(2)

        def matrix(self, pr):
            return self.scale[:, None] * super().matrix(pr)

    circuit = Circuit([CustomRY('a').on(0)])
    operator = _sim(1).get_expectation_with_grad(Hamiltonian(QubitOperator('Z0'), mq.float32), circuit)
    circuit[0].scale[:] = 0.5
    circuit[0].coeff['a'] = 7
    circuit[0].coeff.const = 0.9
    value, gradient = operator([0.3])
    np.testing.assert_allclose(value, [[np.cos(0.3)]], atol=3e-7)
    np.testing.assert_allclose(gradient, [[[-np.sin(0.3)]]], atol=3e-7)
    with pytest.raises(ValueError, match='unitary'):
        _sim(1).get_expectation_with_grad(Hamiltonian(QubitOperator('Z0'), mq.float32), circuit)([0.3])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('t2', [70, 150])
def test_thermal_relaxation_complete_decay_and_identity(t2):
    """Feature: Thermal relaxation trajectories.
    Description: Evolve zero-duration and complete-decay channels for both relaxation regimes.
    Expectation: Zero duration preserves the state and complete decay leaves the ground state.
    """
    state = _state(1)
    sim = _sim(1, state)
    sim.apply_gate(G.ThermalRelaxationChannel(100, t2, 0).on(0))
    np.testing.assert_allclose(sim.get_qs(), state, atol=3e-7)
    sim.apply_gate(G.ThermalRelaxationChannel(100, t2, 1e8).on(0))
    np.testing.assert_allclose(np.abs(sim.get_qs()), [1, 0], atol=3e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_successive_kraus_channels_preserve_normalization_and_relative_phase():
    """
    Feature: Device-resident noise trajectories.
    Description: Apply several Kraus channels separated by coherent evolution in one circuit.
    Expectation: The final pure state matches the analytic reset-and-unitary trajectory.
    """
    sim = _sim(1, _state(1, 429))
    unitary = G.U3(0.71, -0.39, 1.17).matrix()
    circuit = Circuit([G.AmplitudeDampingChannel(1).on(0)]).ry(-0.43, 0)
    circuit += G.KrausChannel('unitary', [unitary]).on(0)
    circuit += G.KrausChannel('phase', [G.S.matrix()]).on(0)
    sim.apply_circuit(circuit)
    expected = G.S.matrix() @ unitary @ G.RY(-0.43).matrix()[:, 0]
    actual = sim.get_qs()
    # Complete decay can retain either branch's global phase, never its norm.
    np.testing.assert_allclose(np.outer(actual, actual.conj()), np.outer(expected, expected.conj()), atol=4e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_noisy_gradient_and_expectation_average_state_dependent_trajectories(monkeypatch):
    """Feature: Stochastic expectation and gradient estimates.
    Description: Average independent amplitude-damping trajectories with an encoder parameter.
    Expectation: Means match the analytic channel, duplicate observables agree and the input is unchanged.
    """
    from mindquantum import _mq_vector_npu as native

    sim = _sim(1, seed=4)
    ham = Hamiltonian(QubitOperator('Z0'), mq.float32)
    circuit = Circuit([G.RY(ParameterResolver({'e': 1.4}).as_encoder()).on(0), G.AmplitudeDampingChannel(0.4).on(0)])
    operator = sim.get_expectation_with_grad([ham, ham], circuit)
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        f, grad = operator(np.full((256, 1), 0.7))
        expectations = [sim.get_expectation(ham, circuit, pr={'e': 0.7}) for _ in range(256)]
    assert f.shape == (256, 2) and grad.shape == (256, 2, 1)
    np.testing.assert_allclose(f[:, 0], f[:, 1], atol=1e-7)
    np.testing.assert_allclose(grad[:, 0], grad[:, 1], atol=1e-7)
    assert f.mean().real == pytest.approx(0.4 + 0.6 * np.cos(0.98), abs=0.08)
    assert np.mean(expectations).real == pytest.approx(0.4 + 0.6 * np.cos(0.98), abs=0.08)
    assert grad.mean().real == pytest.approx(-1.4 * 0.6 * np.sin(0.98), abs=0.15)
    np.testing.assert_array_equal(sim.get_qs(), [1, 0])


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_readonly_expectation_needs_neither_state_copy_nor_download(monkeypatch):
    """Feature: Read-only device expectation.
    Description: Evaluate Hermitian and distinct-bra overlaps without copying or downloading the state.
    Expectation: Both overlaps agree with independent dense-matrix products.
    """
    from mindquantum import _mq_vector_npu as native

    state, other = _state(2, 91), _state(2, 93)
    sim, bra = _sim(2, state), _sim(2, other)
    ham = _ham()
    matrix = ham.hamiltonian.matrix(2).toarray()
    monkeypatch.setattr(native.State, 'copy', _no_download)
    monkeypatch.setattr(native.State, 'get_qs', _no_download)
    np.testing.assert_allclose(sim.get_expectation(ham), np.vdot(state, matrix @ state), atol=2e-7)
    np.testing.assert_allclose(sim.get_expectation(ham, simulator_left=bra), np.vdot(other, matrix @ state), atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n,kept', [(1, []), (4, [3, 1, 2]), (13, [0, 2, 12]), (17, [15]), (17, [0, 2])])
def test_reduced_density_contracts_only_traced_indices(n, kept, monkeypatch):
    """Feature: Device partial trace.
    Description: Retain noncontiguous qubits without copying or downloading the full state.
    Expectation: The result matches NumPy contraction, is Hermitian and preserves trace and input state.
    """
    from mindquantum import _mq_vector_npu as native

    state = _state(n, 104)
    sim = _sim(n, state)
    remaining = sorted(set(range(n)) - set(kept))
    axes = [n - 1 - q for q in sorted(kept, reverse=True)] + [n - 1 - q for q in reversed(remaining)]
    amplitudes = state.reshape([2] * n).transpose(axes).reshape(1 << len(kept), -1)
    expected = amplitudes @ amplitudes.conj().T
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        patch.setattr(native.State, 'copy', _no_download)
        actual = sim.get_reduced_density_matrix(kept)
    np.testing.assert_allclose(actual, expected, atol=2e-7)
    np.testing.assert_allclose(actual, actual.conj().T, atol=1e-7)
    assert np.trace(actual).real == pytest.approx(1, abs=2e-7)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_reduced_state_pure_mixed_and_validation():
    """Feature: Reduced pure and mixed state API.
    Description: Extract entangled and separable subsystems and reject invalid qubit selections.
    Expectation: Mixed subsystems return density matrices and pure subsystems retain their amplitudes.
    """
    sim = _sim(3)
    sim.apply_circuit(Circuit().h(0).x(2, 0).ry(0.31, 1))
    np.testing.assert_allclose(sim.get_qs_of_qubits(0), np.eye(2) / 2, atol=2e-7)
    pure = sim.get_qs_of_qubits(1)
    np.testing.assert_allclose(
        np.outer(pure, pure.conj()), G.RY(0.31).matrix()[:, :1] @ G.RY(0.31).matrix()[:, :1].T, atol=2e-7
    )
    assert isinstance(sim.get_qs_of_qubits([0, 1], ket=True), str)
    for invalid in ([1, 1], [-1], [3], [0, 1, 2]):
        with pytest.raises(ValueError):
            sim.get_reduced_density_matrix(invalid)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('qubits,controls', [([2, 0, 3], []), ([0, 2, 4, 8, 10, 14], [1, 3, 5, 6, 7, 9, 11, 12, 13])])
def test_wide_custom_matrix_and_adjoint(qubits, controls):
    """Feature: Wide controlled custom matrices.
    Description: Embed asymmetric dense operators on noncontiguous targets with internal and external controls.
    Expectation: Evolution and adjoint gradients agree with complex128 references.
    """
    n = max(qubits + controls) + 1
    rng = np.random.default_rng(76)
    dimension = 1 << len(qubits)
    unitary, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension)))
    # Dense zero rows require real zero extension; surrounding untouched bases
    # must retain the identity for evolution, even with sub-eight-lane rows.
    state = _state(n, 78)
    circuit = Circuit().ry('a', qubits[0]) + G.UnivMathGate('wide', unitary).on(qubits, controls)
    sim = _sim(n, state)
    cpu = Simulator('mqvector', n, dtype=mq.complex128)
    cpu.set_qs(state)
    ham = Hamiltonian(QubitOperator(f'Y{qubits[0]}', 0.7) + QubitOperator(f'X{qubits[-1]}', -0.3), mq.float32)
    got = sim.get_expectation_with_grad(ham, circuit)([0.29])
    expected = cpu.get_expectation_with_grad(ham.astype(mq.float64), circuit)([0.29])
    for value, reference in zip(got, expected):
        np.testing.assert_allclose(value, reference, atol=3e-6)
    sim.apply_circuit(circuit, {'a': 0.29})
    np.testing.assert_allclose(sim.get_qs(), _cpu_state(state, circuit, {'a': 0.29}), atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_many_tiles_multiple_observables_and_shared_gradient_plan():
    """Feature: Concurrent public gradient evaluations.
    Description: Evaluate one gradient operator at distinct points while sharing its simulator input.
    Expectation: All observables and gradients match CPU references without changing the input state.
    """
    from concurrent.futures import ThreadPoolExecutor

    n = 14
    state = _state(n, 411)
    circuit = Circuit()
    for i in range(28):
        circuit += G.U3({'a': 0.37}, 0.31 + i / 67, {'b': -0.81}).on(i % n)
        circuit += G.Rxz({'a': -0.53}).on([i % n, (i + 5) % n], [(i + 8) % n])
    circuit += G.RotPauliString('YZXZZ', 'b').on([0, 2, 8, 11, 13], 10)
    hams = [
        Hamiltonian(QubitOperator('Y0 Z13', 0.7) + QubitOperator('X2', -0.9), mq.float32),
        Hamiltonian(QubitOperator('Z1 Z11', -0.43), mq.float32),
    ]
    sim = _sim(n, state)
    cpu = Simulator('mqvector', n, dtype=mq.complex128)
    cpu.set_qs(state)
    op = sim.get_expectation_with_grad(hams, circuit)
    reference = cpu.get_expectation_with_grad([ham.astype(mq.float64) for ham in hams], circuit)
    inputs = [[0.31, -0.79], [-0.61, 0.17]]
    expected = [reference(values) for values in inputs]
    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(op, inputs))
    for got, want in zip(actual, expected):
        for output, target in zip(got, want):
            np.testing.assert_allclose(output, target, atol=4e-6)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n,two_sided', [(12, False), (14, True), (21, False)])
def test_fused_adjoint_shared_controlled_derivatives_and_packet_boundaries(n, two_sided, monkeypatch):
    """Feature: Mixed Cube and Vector adjoint execution.
    Description: Differentiate shared controlled gates across layout changes, packet boundaries and distinct bras.
    Expectation: Repeated results are deterministic and agree with complex128 for both complex gradient components.
    """
    from mindquantum import _mq_vector_npu as native

    # Noncontiguous support, two different packet permutations, partial batches
    # at n=12/14 and multiple batches per AIV at n=21. Derivatives must be ZERO
    # outside controls and phase/exchange support, not the gate's identity.
    circuit = Circuit()
    for target in (1, n - 2):
        for _ in range(3):
            circuit += G.U3({'a': 0.37, 'b': -0.2}, 'b', {'a': -0.71}).on(target, [0, 8])
            circuit += G.FSim('a', {'b': -0.43}).on([4, n - 1], [0, 8])
            circuit += G.RY(ParameterResolver({'a': 0.3, 'b': -0.4}).no_grad_part('b')).on(n - 3)
            circuit += G.PhaseShift({'b': 0.59}).on(target, 8)
            circuit += G.SWAPalpha({'a': -0.29}).on([target, n - 1], 0)
        # Seven logical operands force Vector execution between Cube regions.
        # Both derivatives and folded, no-gradient diagonals see the retained layout.
        circuit += G.RotPauliString('XYZYZXZ', {'a': -0.27}).on([0, 2, 3, 5, 6, 9, n - 1])
        circuit += G.RZ(ParameterResolver({'b': 0.29}).no_grad()).on(6, 2)
        circuit += G.PhaseShift(-0.17).on(n - 1, 2)
        circuit += G.Rzz(0.31).on([0, 5], 2)
    parameters = ParameterResolver({'a': 0.41, 'b': -0.67})
    program = native.AdjointPlan(circuit.get_cpp_obj(), {'a': 0, 'b': 1}, n, fusion_qubits=6).bind(parameters)
    assert program.cube_tile_count >= 2
    assert program.tile_count > program.cube_tile_count
    assert (
        native.AdjointPlan(circuit.get_cpp_obj(), {'a': 0, 'b': 1}, n, cube_min_gates=0)
        .bind(parameters)
        .cube_tile_count
        == 0
    )
    state = _state(n, 607)
    sim = _sim(n, state)
    cpu = Simulator('mqvector', n, dtype=mq.complex128)
    cpu.set_qs(state)
    kwargs, cpu_kwargs = {}, {}
    if two_sided:
        bra = _state(n, 611)
        left = Circuit().ry({'a': -0.31}, 0) + circuit.hermitian()
        left_sim, left_cpu = _sim(n, bra), Simulator('mqvector', n, dtype=mq.complex128)
        left_cpu.set_qs(bra)
        kwargs = {'circ_left': left, 'simulator_left': left_sim}
        cpu_kwargs = {'circ_left': left, 'simulator_left': left_cpu}
    hams = [
        Hamiltonian(QubitOperator(f'Y1 X{n - 1}', 0.71) + QubitOperator('Z4', -0.37), mq.float32),
        Hamiltonian(QubitOperator('', 0.19) + QubitOperator(f'Z0 Z{n - 3}', -0.63), mq.float32),
    ]
    op = sim.get_expectation_with_grad(hams, circuit, **kwargs)
    reference = cpu.get_expectation_with_grad([ham.astype(mq.float64) for ham in hams], circuit, **cpu_kwargs)
    values = [parameters[name] for name in op.ansatz_params_name]
    expected = reference(values)
    if two_sided:
        assert np.max(np.abs(expected[1].imag)) > 1e-4
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        first, second = op(values), op(values)
    for got, repeated, want in zip(first, second, expected):
        np.testing.assert_array_equal(got, repeated)
        np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(sim.get_qs(), state, atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('n', [1, 13])
def test_cached_diagonal_complex_overlap_padding_high_parity_and_reuse(n, monkeypatch):
    """Feature: Diagonal observables and parameter-free gradients.
    Description: Evaluate complex overlaps on distinct states, then rescale one state and reuse the operator.
    Expectation: Identity and high-qubit parity contributions match analytic values, including after rescaling.
    """
    from mindquantum import _mq_vector_npu as native

    state, bra = _state(n, 701), _state(n, 703)
    sim, left = _sim(n, state), _sim(n, bra)
    low = 1 - 2 * ((np.arange(1 << n) >> 0) & 1)
    high = 1 - 2 * ((np.arange(1 << n) >> (n - 1)) & 1)
    first = QubitOperator('', 0.25) - QubitOperator('Z0', 0.125)
    operator = QubitOperator('', 0.5) - QubitOperator('Z0', 2.5)
    operator += QubitOperator(f'Z{n - 1}', 0.75) + 0.375 * QubitOperator('Z0') * QubitOperator(f'Z{n - 1}')
    hams = [Hamiltonian(first, mq.float32), Hamiltonian(operator, mq.float32)]
    expected = [
        np.vdot(bra, (0.25 - 0.125 * low) * state),
        np.vdot(bra, (0.5 - 2.5 * low + 0.75 * high + 0.375 * low * high) * state),
    ]
    circuit = Circuit()
    if n >= 12:
        # A 16-dimensional Cube block with no derivatives must still compute
        # the overlap. These gates commute with both observables, so the
        # independent complex-overlap reference above is unchanged.
        for qubit in (2, 3, 5, 7):
            circuit.h(qubit)
        program = native.AdjointPlan(circuit.get_cpp_obj(), {}, n).bind(ParameterResolver())
        assert program.tile_count == program.cube_tile_count == 1
    op = sim.get_expectation_with_grad(hams, circuit, simulator_left=left)
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        f, g = op([])
        sim.backend.sim.scale(0.25)
        again, _ = op([])
    np.testing.assert_allclose(f[0], expected, atol=3e-7)
    np.testing.assert_allclose(again[0], np.asarray(expected) / 4, atol=1e-7)
    assert g.shape == (1, 2, 0)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_sampling_tree_many_tiles_empty_bins_and_unnormalized_state(monkeypatch):
    """Feature: Terminal sampling from an unnormalized device state.
    Description: Sample a sparse state spanning several tiles with the same seed and reject zero total mass.
    Expectation: Samples stay on occupied indices, follow normalized probabilities and reproduce exactly.
    """
    from mindquantum import _mq_vector_npu as native

    n = 18
    indices = np.array([0, 17, (1 << 17) + 3, (1 << n) - 1])
    probabilities = np.array([0.03, 0.17, 0.29, 0.51])
    state = np.zeros(1 << n, dtype=complex)
    state[indices] = np.sqrt(probabilities) * np.exp(1j * np.arange(4))
    sim = _sim(n, state)
    sim.backend.sim.scale(3)
    with monkeypatch.context() as patch:
        patch.setattr(native.State, 'get_qs', _no_download)
        samples = np.asarray(sim.backend.sim.sample(10000, 99))
        np.testing.assert_array_equal(samples, sim.backend.sim.sample(10000, 99))
    assert set(samples) == set(indices)
    for index, probability in zip(indices, probabilities):
        assert np.mean(samples == index) == pytest.approx(probability, abs=0.02)
    assert sim.backend.sim.norm() == pytest.approx(9, abs=2e-6)
    sim.apply_hamiltonian(Hamiltonian(QubitOperator(), mq.float32))
    with pytest.raises(ValueError, match='positive finite'):
        sim.backend.sim.sample(1, 99)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('qubit', [0, 14])
def test_measurement_rare_branch_reset_preserves_phase(qubit):
    """Feature: Rare measurement outcomes and reset.
    Description: Select a nonzero branch whose probability is below FP32 subtraction resolution.
    Expectation: The branch remains normalizable and reset preserves its relative phase.
    """
    state = np.zeros(1 << 15, dtype=complex)
    state[0] = np.sqrt(1 - 1e-12)
    state[(1 << qubit) | 18] = 1e-6j
    sim = _sim(15, state)
    # Force a rare, nonzero branch. Computing it as total minus the dominant
    # branch in FP32 incorrectly gives zero and cannot normalize this outcome.
    assert sim.backend.sim.measure(qubit, 0.0, 0) == 1
    expected = np.zeros_like(state)
    expected[18] = 1j
    np.testing.assert_allclose(sim.get_qs(), expected, atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_csr_long_duplicate_rows_empty_operator_and_asymmetric_bra():
    """Feature: General CSR and empty observables.
    Description: Apply long rows with duplicate columns and evaluate distinct-bra overlaps and empty operators.
    Expectation: Results match SciPy multiplication and empty observables produce exact zeros.
    """
    n = 11
    dimension = 1 << n
    columns = (7 * np.arange(1103) + 53) % 1024
    values = (0.31 * np.exp(1j * np.arange(1103) / 13)).astype(np.complex64)
    offsets = np.zeros(dimension + 1, dtype=np.int32)
    offsets[4:] = len(values)
    sparse = csr_matrix((values, columns, offsets), shape=(dimension, dimension))
    state, bra = _state(n, 551), _state(n, 553)
    sim, left = _sim(n, state), _sim(n, bra)
    ham = Hamiltonian(sparse, mq.complex64)
    expected = sparse.astype(np.complex128) @ state
    np.testing.assert_allclose(sim.get_expectation(ham, simulator_left=left), np.vdot(bra, expected), atol=2e-7)
    sim.apply_hamiltonian(ham)
    np.testing.assert_allclose(sim.get_qs(), expected, atol=8e-7)
    for empty in (
        Hamiltonian(QubitOperator(), mq.float32),
        Hamiltonian(csr_matrix((dimension, dimension)), mq.complex64),
    ):
        sim.set_qs(state)
        assert sim.get_expectation(empty) == 0
        sim.apply_hamiltonian(empty)
        np.testing.assert_array_equal(sim.get_qs(), np.zeros(dimension))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_native_parameter_validation_and_late_expression_errors():
    """Feature: Finite real parameter boundaries.
    Description: Reject complex, nonfinite and overflowing expressions through native and Python entry points.
    Expectation: Invalid inputs fail before state or random-generator mutation, including unused resolver values.
    """
    from mindquantum import _mq_vector_npu as native

    sim = _sim(2, _state(2, 83))
    before = sim.get_qs()
    control = sim.copy()
    for coefficient in (1j, np.inf, np.nan):
        gate = G.RY(ParameterResolver({'a': coefficient})).on(1)
        circuit = Circuit().h(0).measure('before_invalid', 0) + gate
        with pytest.raises(ValueError, match='real and finite'):
            sim.apply_circuit(circuit, {'a': 0.0})
        with pytest.raises(ValueError, match='real and finite'):
            native.ForwardPlan([gate.get_cpp_obj()], 2)
    with pytest.raises(ValueError, match='finite'):
        sim.apply_circuit(Circuit().h(0).ry({'a': 1e308}, 1), {'a': 1e308})
    plan = native.ForwardPlan([G.RX('a').on(0).get_cpp_obj()], 2)
    for value in (1j, np.inf, np.nan):
        with pytest.raises(ValueError, match='real and finite'):
            plan.bind(ParameterResolver({'a': 0.2, 'unused': value}))
    np.testing.assert_array_equal(sim.get_qs(), before)
    probe = Circuit().h(0).measure('coin', 0)
    np.testing.assert_array_equal(sim.sampling(probe, shots=64).samples, control.sampling(probe, shots=64).samples)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('cube', [0, 4])
def test_native_plan_zero_crossings_batches_and_independent_concurrent_states(cube):
    """Feature: Native plan ownership and concurrent independent states.
    Description: Reuse a plan after deleting its creator and vary active and frozen parameters across zero.
    Expectation: Concurrent batches match CPU references and reuse cannot leak state, parameters or partial sums.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mindquantum import _mq_vector_npu as native

    n = 13
    circuit = Circuit([G.H.on(q) for q in (0, 3, 8, 12)])
    for q in (1, 9, 2, 11):
        circuit += G.U3({'a': 0.37}, {'b': -0.71}, 'a').on(q, 4)
        circuit += G.FSim('b', {'a': -0.53}).on([q, 7], 4)
    circuit += G.RotPauliString('YZXZZZ', 'a').on([0, 3, 5, 8, 10, 12], 6)
    # no_grad is an input dependency, not a compile-time constant. Deliberately
    # omit it from the native output map while changing its value between calls.
    circuit += G.RY(ParameterResolver('frozen').no_grad()).on(10, 12)
    hams = [_ham(), Hamiltonian(QubitOperator('Y7 Z12', 0.43) + QubitOperator('Z4', -0.29), mq.float32)]
    states = [_state(n, 919), _state(n, 923)]
    sims = [_sim(n, state) for state in states]
    owner = _sim(n)
    plan = native.make_gradient(
        owner.backend.sim,
        [ham.get_cpp_obj() for ham in hams],
        circuit.get_cpp_obj(),
        [],
        ['a', 'b'],
        True,
        12,
        6,
        cube,
    )
    points = [{'a': 0.0, 'b': 0.0, 'frozen': 0.0}, {'a': 0.47, 'b': -0.83, 'frozen': 0.62}]
    parameters = [ParameterResolver(point) for point in (*points, points[0])]
    expected = []
    for state in states:
        cpu = Simulator('mqvector', n, dtype=mq.complex128)
        cpu.set_qs(state)
        reference = cpu.get_expectation_with_grad([ham.astype(mq.float64) for ham in hams], circuit)
        batch = []
        for point in (*points, points[0]):
            value, gradient = reference([point[name] for name in circuit.params_name])
            batch.append(np.concatenate([value[..., None], gradient[..., :2]], axis=-1)[0])
        expected.append(batch)
    # The native program owns its snapshot/context independently of its creator.
    circuit[4].get_parameters()[0].const = 2.1
    del owner, circuit, hams
    gc.collect()

    def evaluate(sim):
        state = sim.backend.sim
        return np.asarray(plan.evaluate(state, state, parameters))

    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(evaluate, sims))
    for sim, initial, got, want in zip(sims, states, actual, expected):
        np.testing.assert_allclose(got, want, atol=4e-6, rtol=2e-5)
        np.testing.assert_array_equal(got[0], got[2])
        # A completed call may lend storage to a different input state; no
        # partial sum, layout, or zero-sized metadata image may survive binding.
        np.testing.assert_array_equal(evaluate(sim), got)
        np.testing.assert_allclose(sim.get_qs(), initial, atol=2e-7)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('cube', [0, 4])
@pytest.mark.parametrize('n,width,parameters', [(12, 4, 3), (13, 5, 5), (21, 6, 9), (17, 4, 275)])
def test_wide_shared_parameter_gradients_across_state_tiles(cube, n, width, parameters):
    """
    Feature: Adjoint gradients across operator sizes and shared parameter counts.
    Description: Reuse affine expressions across controlled gates, partial packets and multiple state tiles.
    Expectation: Every output agrees with complex128 CPU evolution, including on a repeated call.
    """
    target = n - 1
    first = {f'p{i:03}': (-1) ** i * (0.07 + i / 1900) for i in range(parameters)}
    second = {name: 0.19 - i / 1300 for i, name in enumerate(reversed(first))}
    circuit = Circuit()
    for _ in range(2):
        circuit += G.RY(first).on(target)
        circuit += G.RZ(-0.37).on(3)
        circuit += G.X.on(3, [target] + [0, 1, 2, 4][: width - 2])
        circuit += G.RX(second).on(3)
        circuit += G.PhaseShift(0.23).on(target)
    # Coherent inputs keep gradients finite as spectator qubits are added;
    # a large Haar-random state could hide lost contributions under atol.
    state = np.ones(1, dtype=complex)
    for qubit in range(n):
        angle = 0.41 + 0.73 * (qubit % 5) / 4
        state = np.kron([np.cos(angle), np.sin(angle) * np.exp(1j * (0.19 + qubit / 13))], state)
    sim = Simulator(MQVectorNPU(n, cube_min_gates=cube))
    cpu = Simulator('mqvector', n, dtype=mq.complex128)
    sim.set_qs(state)
    cpu.set_qs(state)
    ham = Hamiltonian(QubitOperator(f'Y3 Z{target}', -0.71) + QubitOperator(f'X{target}', 0.29), mq.float32)
    op = sim.get_expectation_with_grad(ham, circuit)
    reference = cpu.get_expectation_with_grad(ham.astype(mq.float64), circuit)
    for values in (np.linspace(-0.41, 0.73, len(first)), np.zeros(len(first))):
        actual, expected = op(values), reference(values)
        assert actual[1].shape == (1, 1, len(first))
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-5)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('cube', [0, 4])
def test_constant_adjoint_blocks_preserve_parity_ladder_gradient(cube):
    """
    Feature: Gradients through constant blocks spanning multiple packet batches.
    Description: Conjugate a parameterized phase by a parity ladder and fixed basis changes.
    Expectation: The energy and gradient follow cosine and minus sine without losing amplitudes.
    """
    circuit = Circuit([G.X.on(qubit) for qubit in range(12)])
    circuit.rx(np.pi / 2, 0).h(12)
    for qubit in range(12):
        circuit.x(qubit + 1, qubit)
    circuit.rz('theta', 12)
    for qubit in reversed(range(12)):
        circuit.x(qubit + 1, qubit)
    circuit.h(12).rx(-np.pi / 2, 0)
    sim = Simulator(MQVectorNPU(21, cube_min_gates=cube))
    op = sim.get_expectation_with_grad(Hamiltonian(QubitOperator('Z12', 0.7), mq.float32), circuit)
    for theta in (0.37, -0.63, 0.0):
        energy, gradient = op([theta])
        np.testing.assert_allclose(energy, [[0.7 * np.cos(theta)]], atol=2e-6)
        np.testing.assert_allclose(gradient, [[[-0.7 * np.sin(theta)]]], atol=2e-6)
