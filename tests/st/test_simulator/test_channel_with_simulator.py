# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Noise-channel contracts: exact process states first, sampling only for trajectory weights."""

from functools import reduce
from itertools import product

import numpy as np
import pytest
from scipy.linalg import hadamard
from scipy.stats import binom

import mindquantum as mq
from mindquantum.core import gates as G  # noqa: N812
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator, get_supported_simulator

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu, pytest.mark.env_onecard]

# The mqvector family implements general noise trajectories; mqmps and stabilizer do not.
_VECTORS = [c for c in get_supported_simulator() if c != 'stabilizer' and c[0].startswith('mqvector')]
_SEED = 42
_SHOTS = 1024

_MIXED_CHANNELS = (
    G.PauliChannel(0.06, 0.24, 0.52),
    G.DepolarizingChannel(4 / 3),
    G.DepolarizingChannel(0.48, 2),
    G.DepolarizingChannel(64 / 63, 3),  # No identity Pauli string at the upper bound.
    G.GroupedPauliChannel(np.array([[0.8, 0.0, 0.0], [0.0, 0.35, 0.0], [0.0, 0.0, 0.6]])),
    G.AmplitudeDampingChannel(0.72),
    G.PhaseDampingChannel(0.84),
    G.KrausChannel('rotated_damping', [G.S.matrix() @ k for k in G.AmplitudeDampingChannel(0.64).matrix()]),
    G.KrausChannel('phased_paulis', [1j * k for k in G.PauliChannel(0.11, 0.19, 0.53).matrix()]),
    G.ThermalRelaxationChannel(100, 35, 90),
    G.ThermalRelaxationChannel(100, 125, 90),
)
# The flip wrappers use the Pauli sampler above. Rank-one cases also fix the entire
# trajectory (up to an unobservable global phase), at either backend precision.
_EXACT_CHANNELS = (
    *_MIXED_CHANNELS,
    G.BitFlipChannel(0.3),
    G.BitPhaseFlipChannel(0.6),
    G.PhaseFlipChannel(0.7),
    G.BitFlipChannel(1),
    G.BitPhaseFlipChannel(1),
    G.PhaseFlipChannel(1),
    G.AmplitudeDampingChannel(1),
    G.PhaseDampingChannel(0),
    G.ThermalRelaxationChannel(100, 125, 0),
    G.KrausChannel('complex_unitary', [G.U3(0.8, -0.4, 1.2).matrix()]),
)


def _process_state(channel, amplitudes):
    """NumPy output for sum_i a_i |i>_sys |i>_ref; the channel acts on the high bits."""
    dim = len(amplitudes)
    if isinstance(channel, G.DepolarizingChannel):
        state = np.diag(amplitudes).ravel()
        mixed = np.kron(np.eye(dim) / dim, np.diag(np.abs(amplitudes) ** 2))
        return (1 - channel.p) * np.outer(state, state.conj()) + channel.p * mixed
    if isinstance(channel, G.GroupedPauliChannel):
        # A product of independent one-qubit channels, not a second noisy simulation.
        local = [G.PauliChannel(*row).matrix() for row in reversed(channel.probs)]
        kraus = (reduce(np.kron, factors) for factors in product(*local))
    else:
        kraus = channel.matrix()
    # The (system, reference) amplitude after K is K[system, reference] * a[reference].
    branches = np.stack([(k * amplitudes).ravel() for k in kraus])
    return branches.T @ branches.conj()


@pytest.fixture(name='choi_probe', scope='module', params=_EXACT_CHANNELS, ids=str)
def _choi_probe(request):
    """One maximally entangled input characterizes the channel on all inputs."""
    channel = request.param
    dim = 1 << channel.n_qubits
    amplitudes = np.full(dim, 1 / np.sqrt(dim))
    gate = channel.on(range(channel.n_qubits, 2 * channel.n_qubits))
    return gate, np.diag(amplitudes).ravel(), _process_state(channel, amplitudes)


@pytest.mark.parametrize('dtype', [mq.complex64, mq.complex128])
def test_density_channel_choi(choi_probe, dtype):
    """
    Feature: Density-matrix noise channel evolution.
    Description: Apply each supported channel to half of a maximally entangled state on mqmatrix.
    Expectation: Output density matrix matches the normalized Choi matrix including coherences.
    """
    gate, state, expected = choi_probe
    sim = Simulator('mqmatrix', 2 * gate.n_qubits, seed=_SEED, dtype=dtype)
    sim.set_qs(state)
    sim.apply_gate(gate)
    np.testing.assert_allclose(sim.get_qs(), expected, rtol=0, atol=1e-6 if dtype == mq.complex64 else 1e-12)


@pytest.mark.parametrize('backend,dtype', _VECTORS)
def test_channel_trajectory_support(choi_probe, backend, dtype):
    """
    Feature: State-vector noise trajectory validity.
    Description: Evolve a maximally entangled input through one noisy trajectory on each vector backend.
    Expectation: Output state has unit norm and zero projection onto the kernel of the Choi state.
    """
    gate, state, expected = choi_probe
    sim = Simulator(backend, 2 * gate.n_qubits, seed=_SEED, dtype=dtype)
    sim.set_qs(state)
    sim.apply_gate(gate)
    actual = sim.get_qs()
    atol = 1e-6 if dtype == mq.complex64 else 1e-12
    np.testing.assert_allclose(np.linalg.norm(actual), 1, rtol=0, atol=atol)
    # Support, unlike membership in a particular list of Kraus branches, is
    # invariant under a change of unraveling (e.g. thermal relaxation on Ascend).
    weights, basis = np.linalg.eigh(expected)
    np.testing.assert_allclose(basis[:, weights < 1e-12].conj().T @ actual, 0, rtol=0, atol=atol)


def _bell_probabilities(rho, dim):
    """CNOTs map |s,r> to |s xor r,r>; Hadamards read the phase on the reference register."""
    system, reference = np.divmod(np.arange(dim**2), dim)
    permutation = (system ^ reference) * dim + reference
    readout = np.kron(np.eye(dim), hadamard(dim) / np.sqrt(dim))[:, permutation]
    return np.clip(np.diag(readout @ rho @ readout.T).real, 0, 1)  # Roundoff at the boundary.


@pytest.mark.parametrize('backend', sorted({name for name, _ in _VECTORS}))
@pytest.mark.parametrize('channel', _MIXED_CHANNELS, ids=str)
def test_trajectory_weights_in_bell_basis(backend, channel):
    """
    Feature: State-vector noise trajectory sampling weights.
    Description: Sample entangled states in the Bell basis after applying each mixed noise channel.
    Expectation: Sampled syndrome counts fall within exact binomial prediction intervals.
    """
    n = channel.n_qubits
    dim = 1 << n
    if isinstance(channel, (G.PauliChannel, G.GroupedPauliChannel, G.DepolarizingChannel)):
        # Each Pauli string gives a distinct, deterministic Bell syndrome.
        amplitudes = np.full(dim, 1 / np.sqrt(dim))
    else:
        # Unequal populations probe state-dependent jump rates; the phase makes
        # the readout sensitive to both real and imaginary transfer coefficients.
        amplitudes = np.sqrt([0.2, 0.8]) * [1, np.exp(0.5j)]
    probabilities = _bell_probabilities(_process_state(channel, amplitudes), dim)
    sim = Simulator(backend, 2 * n, seed=_SEED)
    sim.set_qs(np.diag(amplitudes).ravel())
    circuit = Circuit(channel.on(range(n, 2 * n)))
    for q in range(n):
        circuit.x(n + q, q).h(q)
    result = sim.sampling(circuit.measure_all(), shots=_SHOTS, seed=_SEED)

    # Each multinomial bin is marginally binomial. Exact two-sided prediction
    # limits work for rare and impossible syndromes too. With <200 bins across
    # the suite the false rejection probability is <2e-6 per backend (union bound).
    low, high = binom.interval(1 - 1e-8, _SHOTS, probabilities)
    assert result.shots == _SHOTS
    for i, (lo, hi) in enumerate(zip(low, high)):
        key = format(i, f'0{2 * n}b')
        count = result.data.get(key, 0)
        assert lo <= count <= hi, f'{key}: {count}/{_SHOTS}, expected p={probabilities[i]:.5f}, counts in [{lo}, {hi}]'
