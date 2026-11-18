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
# ============================================================================
"""Regression tests for non-finite simulator inputs.

Reported by mindquantum issues:

* `#464 <https://gitcode.com/mindspore/mindquantum/issues/464>`_: the CPU
  ``apply_circuit``/``apply_gate`` path accepts ``NaN``/``Inf`` parameter values
  and silently poisons the quantum state.  The ``mqvector_npu`` backend already
  rejects the same input with a ``ValueError``, so the two backends disagree.
* `#465 <https://gitcode.com/mindspore/mindquantum/issues/465>`_: the CPU
  ``set_qs`` path accepts ``NaN``/``Inf`` state vectors and, for an overflowing
  vector such as ``1e300``, silently normalizes it to the zero state.

The tests below describe the behaviour the two CPU paths are expected to share
with the Ascend path: validate the finiteness of the user supplied data first,
raise ``ValueError`` before touching the simulator state, and keep the previous
state intact when the call is rejected.

Note:
    The three failure cases of the two issues are collected in one test per
    backend and per API entry point.  The affected checks are marked ``xfail``
    (``strict=False``) so that the level-0 CI stays green while issues #464 and
    #465 are open; they become ordinary regression tests once
    ``_set_vector_state``/``_set_matrix_state`` and the parameter resolver of
    ``apply_circuit``/``apply_gate`` validate ``np.isfinite``.  Remove the
    markers together with the corresponding fix.
"""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum import Circuit
from mindquantum.core.gates import RX, RZ
from mindquantum.simulator import Simulator

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]

_BACKENDS = ['mqvector', 'mqmatrix']
_NONFINITE = [np.nan, np.inf, -np.inf]

_ISSUE_464 = pytest.mark.xfail(
    reason='Open issue #464: apply_circuit/apply_gate do not validate non-finite parameters yet.',
    strict=False,
)
_ISSUE_465 = pytest.mark.xfail(
    reason='Open issue #465: set_qs does not validate non-finite states yet.',
    strict=False,
)


def _new_simulator(backend, n_qubits=4):
    """Create a deterministic simulator in the ground state."""
    return Simulator(backend, n_qubits, seed=7)


def _assert_unchanged(simulator, expected):
    """Check that a rejected call left the quantum state untouched."""
    np.testing.assert_allclose(simulator.get_qs(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('backend', _BACKENDS)
@_ISSUE_465
def test_set_qs_rejects_nonfinite_state_vector(backend):
    """
    Feature: Input validation of ``Simulator.set_qs``.

    Description: Set a state vector filled with ``NaN`` or ``+/-Inf`` and inspect the stored state.

    Expectation: A ``ValueError`` about non-finite values is raised and the previous state is kept.
    """
    simulator = _new_simulator(backend, 3)
    before = simulator.get_qs()
    for value in _NONFINITE:
        with pytest.raises(ValueError, match='finite'):
            simulator.set_qs(np.full(1 << 3, value))
        _assert_unchanged(simulator, before)


@pytest.mark.parametrize('backend', _BACKENDS)
@_ISSUE_465
def test_set_qs_rejects_overflowing_state_vector(backend):
    """
    Feature: Input validation of ``Simulator.set_qs``.

    Description: Set a state vector whose squared norm overflows the float64 range.

    Expectation: A ``ValueError`` is raised instead of silently storing the zero state.
    """
    simulator = _new_simulator(backend, 3)
    before = simulator.get_qs()
    with pytest.raises(ValueError, match='finite|Wrong quantum state'):
        simulator.set_qs(np.full(1 << 3, 1e300))
    _assert_unchanged(simulator, before)


@pytest.mark.parametrize(
    'bad_diagonal',
    [
        pytest.param(np.inf, id='inf'),
        pytest.param(1e308, id='overflowing-trace'),
    ],
)
@_ISSUE_465
def test_set_qs_rejects_invalid_density_matrix(bad_diagonal):
    """
    Feature: Input validation of ``Simulator.set_qs`` for density matrices.

    Description: Set a Hermitian density matrix whose entries or trace are not representable.

    Expectation: A ``ValueError`` is raised and the previous state is kept.
    """
    simulator = _new_simulator('mqmatrix', 2)
    before = simulator.get_qs()
    matrix = np.diag(np.full(1 << 2, bad_diagonal, dtype=np.float64))
    with pytest.raises(ValueError, match='finite|Wrong quantum state|trace'):
        simulator.set_qs(matrix)
    _assert_unchanged(simulator, before)


@pytest.mark.parametrize('backend', _BACKENDS)
@_ISSUE_464
def test_apply_circuit_rejects_nonfinite_parameters(backend):
    """
    Feature: Input validation of ``Simulator.apply_circuit``.

    Description: Apply a parameterized circuit with a ``NaN``/``Inf`` parameter value.

    Expectation: A ``ValueError`` about non-finite values is raised and the previous state is kept.
    """
    circuit = Circuit().h(0).rx('x', 1)
    for value in _NONFINITE:
        simulator = _new_simulator(backend)
        before = simulator.get_qs()
        with pytest.raises(ValueError, match='finite'):
            simulator.apply_circuit(circuit, {'x': value})
        _assert_unchanged(simulator, before)


@pytest.mark.parametrize('backend', _BACKENDS)
@_ISSUE_464
def test_apply_circuit_rejects_nonfinite_parameter_resolver(backend):
    """
    Feature: Input validation of ``Simulator.apply_circuit``.

    Description: Pass the non-finite parameter through a ``ParameterResolver`` object.

    Expectation: A ``ValueError`` about non-finite values is raised and the previous state is kept.
    """
    circuit = Circuit().ry('theta', 0).rz('phi', 1)
    for value in _NONFINITE:
        simulator = _new_simulator(backend)
        before = simulator.get_qs()
        with pytest.raises(ValueError, match='finite'):
            simulator.apply_circuit(circuit, mq.ParameterResolver({'theta': 0.1, 'phi': value}))
        _assert_unchanged(simulator, before)


@pytest.mark.parametrize('backend', _BACKENDS)
@_ISSUE_464
def test_apply_gate_rejects_nonfinite_concrete_parameter(backend):
    """
    Feature: Input validation of ``Simulator.apply_gate``.

    Description: Apply a gate whose parameter has been materialized to ``NaN`` or ``Inf``.

    Expectation: A ``ValueError`` about non-finite values is raised and the previous state is kept.
    """
    for value in _NONFINITE:
        simulator = _new_simulator(backend)
        before = simulator.get_qs()
        with pytest.raises(ValueError, match='finite'):
            simulator.apply_gate(RX(value).on(1))
        _assert_unchanged(simulator, before)


@pytest.mark.parametrize('backend', _BACKENDS)
@pytest.mark.parametrize('gate_type', [RX, RZ])
@_ISSUE_464
def test_apply_gate_rejects_nonfinite_parameter_resolver(backend, gate_type):
    """
    Feature: Input validation of ``Simulator.apply_gate``.

    Description: Apply a symbolic single-qubit rotation resolved with a non-finite value.

    Expectation: A ``ValueError`` about non-finite values is raised and the previous state is kept.
    """
    for value in _NONFINITE:
        simulator = _new_simulator(backend)
        before = simulator.get_qs()
        with pytest.raises(ValueError, match='finite'):
            simulator.apply_gate(gate_type('theta').on(1), {'theta': value})
        _assert_unchanged(simulator, before)


@pytest.mark.parametrize('backend', _BACKENDS)
@_ISSUE_464
def test_sampling_after_rejected_update_is_still_valid(backend):
    """
    Feature: State consistency after a rejected parameter update.

    Description: Try a non-finite update, then run a normal circuit, an expectation and a sampling.

    Expectation: The simulator recovers and returns finite results.
    """
    simulator = _new_simulator(backend, 2)
    with pytest.raises(ValueError, match='finite'):
        simulator.apply_gate(RX(np.nan).on(0))
    simulator.apply_circuit(Circuit().h(0))
    expectation = simulator.get_expectation(mq.Hamiltonian(mq.QubitOperator('Z0')), Circuit().ry('a', 1), pr={'a': 0.2})
    assert np.isfinite(expectation).all()
    result = simulator.sampling(Circuit().measure('m', 0), shots=10, seed=3)
    assert sum(result.data['m'].values()) == 10


@pytest.mark.parametrize('backend', _BACKENDS)
def test_finite_extreme_parameters_are_still_accepted(backend):
    """
    Feature: Input validation of ``Simulator.apply_circuit`` and ``Simulator.apply_gate``.

    Description: Use large but finite parameter values after the finiteness validation is added.

    Expectation: The gates are applied and the resulting state stays finite and normalized.
    """
    simulator = _new_simulator(backend)
    simulator.apply_circuit(Circuit().rx('x', 0), {'x': 1e6})
    simulator.apply_gate(RZ(-1e6).on(1))
    state = simulator.get_qs()
    assert np.isfinite(state).all()
    if backend == 'mqvector':
        assert np.isclose(np.linalg.norm(state), 1.0)
    else:
        assert np.isclose(np.trace(state), 1.0)


def test_set_qs_still_accepts_valid_states():
    """
    Feature: Input validation of ``Simulator.set_qs``.

    Description: Set a normalized ket and a valid density matrix on the two CPU backends.

    Expectation: Both states are stored exactly as provided.
    """
    ket = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.complex128)
    ket /= np.linalg.norm(ket)
    vector_simulator = _new_simulator('mqvector', 2)
    vector_simulator.set_qs(ket)
    np.testing.assert_allclose(vector_simulator.get_qs(), ket)

    density_simulator = _new_simulator('mqmatrix', 2)
    density_simulator.set_qs(ket)
    np.testing.assert_allclose(density_simulator.get_qs(), np.outer(ket, ket.conj()))

    density_matrix = np.diag([0.25, 0.75, 0.0, 0.0]).astype(np.complex128)
    density_simulator.set_qs(density_matrix)
    np.testing.assert_allclose(density_simulator.get_qs(), density_matrix)
