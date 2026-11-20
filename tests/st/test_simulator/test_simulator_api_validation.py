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
"""Simulator boundary checks before dispatch to native kernels."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum import Circuit
from mindquantum.core.gates import RX, AmplitudeDampingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


def _simulator(backend='mqvector', n_qubits=2, dtype=mq.complex128):
    return Simulator(backend, n_qubits, dtype=dtype)


_HAMILTONIAN = Hamiltonian(QubitOperator('Z0'))
_ROTATION = Circuit().ry('a', 0)
_NOISY = _ROTATION + AmplitudeDampingChannel(0.1).on(0)


@pytest.mark.parametrize(
    'call,error,match',
    [
        pytest.param(lambda s: s.sampling(Circuit().h(0)), ValueError, 'measurement', id='sample-no-measurement'),
        pytest.param(
            lambda s: s.sampling(Circuit().measure('m', 0), shots=1.5),
            TypeError,
            'requires an int',
            id='sample-shots-type',
        ),
        pytest.param(
            lambda s: s.sampling(_ROTATION + Circuit().measure('m', 0), pr=1.5),
            TypeError,
            'requires a dict or a ParameterResolver',
            id='sample-parameters-type',
        ),
        pytest.param(
            lambda s: s.sampling(Circuit().measure('m', 0), shots=0),
            ValueError,
            'not less than 1',
            id='sample-zero-shots',
        ),
        pytest.param(
            lambda s: s.sampling(_ROTATION + Circuit().measure('m', 0)),
            ValueError,
            'ParameterResolver',
            id='sample-unbound',
        ),
        pytest.param(
            lambda s: s.sampling(Circuit().measure('m', 2)), ValueError, 'more than simulator qubits', id='sample-width'
        ),
        pytest.param(lambda s: s.set_qs(np.ones(3)), ValueError, 'power of 2', id='state-non-qubit-size'),
        pytest.param(lambda s: s.set_qs(np.zeros(4)), ValueError, 'Wrong quantum state', id='state-zero-norm'),
        pytest.param(
            lambda s: _simulator('mqmatrix', 1).set_qs(np.zeros((2, 2))),
            ValueError,
            'Wrong quantum state',
            id='state-zero-trace',
        ),
        pytest.param(lambda s: s.get_qs(ket='yes'), TypeError, 'requires a bool', id='state-ket-type'),
        pytest.param(lambda s: s.set_qs(np.ones(2)), ValueError, 'simulation qubits', id='state-width'),
        pytest.param(lambda s: s.set_qs(np.eye(4)), ValueError, '1-dimensional', id='vector-rank'),
        pytest.param(
            lambda s: _simulator('mqmatrix', 1).set_qs(np.zeros((2, 2, 2))),
            ValueError,
            '1D vector or 2D',
            id='matrix-rank',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix', 1).set_qs(np.array([[0, 1], [0, 0]])),
            ValueError,
            'hermitian',
            id='state-non-hermitian',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix', 1).set_qs(np.diag([-0.5, 1.5])),
            ValueError,
            'negative',
            id='state-negative-population',
        ),
        pytest.param(
            lambda s: s.apply_circuit(Circuit().h(2)), ValueError, 'more than simulator qubits', id='circuit-width'
        ),
        pytest.param(lambda s: s.apply_circuit(_ROTATION), ValueError, 'parameter_resolver', id='circuit-unbound'),
        pytest.param(
            lambda s: s.apply_circuit(_ROTATION, {'b': 1}), ValueError, 'not in given', id='circuit-wrong-parameter'
        ),
        pytest.param(
            lambda s: s.apply_gate(RX(0.1).on(2)), ValueError, 'higher than simulator qubits', id='gate-width'
        ),
        pytest.param(lambda s: s.apply_gate(RX('a').on(0)), ValueError, 'parameter_resolver', id='gate-unbound'),
        pytest.param(
            lambda s: s.get_expectation(QubitOperator('Z0')),
            TypeError,
            'requires a Hamiltonian',
            id='expectation-operator-type',
        ),
        pytest.param(
            lambda s: Simulator('stabilizer', 2).get_expectation(_HAMILTONIAN, _ROTATION, pr={'a': 0.1}),
            ValueError,
            'only supports hamiltonian and circ_right',
            id='stabilizer-expectation-options',
        ),
        pytest.param(
            lambda s: s.get_expectation(Hamiltonian(QubitOperator('Z2'))),
            ValueError,
            'bigger than simulator qubits',
            id='expectation-width',
        ),
        pytest.param(
            lambda s: s.get_expectation(Hamiltonian(QubitOperator('Z0'), mq.complex64)),
            TypeError,
            'Data type',
            id='expectation-precision',
        ),
        pytest.param(
            lambda s: s.get_expectation(_HAMILTONIAN, simulator_left=_simulator(dtype=mq.complex64)),
            TypeError,
            'different with simulator_left',
            id='bra-precision',
        ),
        pytest.param(
            lambda s: s.get_expectation(_HAMILTONIAN, simulator_left=_simulator('mqmatrix')),
            ValueError,
            'same type',
            id='bra-backend',
        ),
        pytest.param(
            lambda s: s.get_expectation(_HAMILTONIAN, simulator_left=_simulator(n_qubits=3)),
            ValueError,
            'Require a 2 simulator',
            id='bra-width',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_expectation(_HAMILTONIAN, _ROTATION, _ROTATION),
            NotImplementedError,
            'Non hermitian',
            id='density-bra-ket',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(1, _ROTATION),
            TypeError,
            'requires a Hamiltonian or a list',
            id='gradient-operator-type',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_expectation_with_grad(_HAMILTONIAN, _ROTATION, _ROTATION),
            ValueError,
            "doesn't support circ_left",
            id='density-left-gradient',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_expectation_with_grad(_HAMILTONIAN, _ROTATION, pr_shift=True),
            ValueError,
            'parameter-shift',
            id='density-parameter-shift',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, Circuit().rx('a', 0).as_encoder().rx('a', 1)),
            RuntimeError,
            'already set to encoder',
            id='conflicting-parameter-roles',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, Circuit().measure('m', 0)),
            ValueError,
            'measure gate',
            id='gradient-measurement',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, Circuit().ry('a', 2)),
            ValueError,
            'circuit has 3 qubits',
            id='gradient-width',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(Hamiltonian(QubitOperator('Z0'), mq.complex64), _ROTATION),
            TypeError,
            'single precision',
            id='gradient-precision',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, _ROTATION, simulator_left=_simulator('mqmatrix')),
            ValueError,
            'same backend',
            id='gradient-bra-backend',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, _ROTATION, simulator_left=_simulator(n_qubits=3)),
            ValueError,
            'same n_qubits',
            id='gradient-bra-width',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, _NOISY, circ_left=_ROTATION),
            ValueError,
            'noise circuit use parameter shift',
            id='noisy-left-circuit',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, _NOISY, simulator_left=_simulator()),
            ValueError,
            'noise circuit use parameter shift',
            id='noisy-left-simulator',
        ),
        pytest.param(
            lambda s: s.get_expectation_with_grad(_HAMILTONIAN, _ROTATION, circ_left=_NOISY),
            ValueError,
            'noise left circuit',
            id='noisy-bra',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_partial_trace([0, 0]), ValueError, 'repeated', id='trace-duplicate'
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_partial_trace(2),
            ValueError,
            'higher than simulator qubits',
            id='trace-range',
        ),
        pytest.param(lambda s: s.get_partial_trace(0), ValueError, 'not support partial trace', id='trace-vector'),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_partial_trace(1.5), TypeError, 'requires a', id='trace-qubits-type'
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_qs_of_qubits([0, 1.5]),
            TypeError,
            'must be integers',
            id='reduced-state-qubit-type',
        ),
        pytest.param(
            lambda s: _simulator('mqmatrix').get_reduced_density_matrix([0, 1]),
            ValueError,
            'less than total',
            id='reduction-all-qubits',
        ),
        pytest.param(
            lambda s: s.apply_hamiltonian(Hamiltonian(QubitOperator('Z0'), mq.complex64)),
            TypeError,
            'Data type',
            id='apply-hamiltonian-precision',
        ),
        pytest.param(lambda s: s.astype(mq.float32), TypeError, 'complex type', id='astype-real'),
    ],
)
def test_invalid_simulator_request(call, error, match):
    """
    Feature: Simulator API validation.
    Description: Submit an invalid state, operation or expectation request.
    Expectation: The violated contract raises an explicit error before native execution.
    """
    with pytest.raises(error, match=match):
        call(_simulator())


@pytest.mark.parametrize('method', ['apply_gate', 'apply_circuit'])
def test_complex_parameter_warning(method):
    """
    Feature: Complex rotation parameters.
    Description: Apply a rotation with a non-zero imaginary parameter component.
    Expectation: A warning is emitted and evolution uses only the real component.
    """
    operations = {'apply_gate': RX('a').on(0), 'apply_circuit': Circuit().rx('a', 0)}
    simulator = _simulator()
    with pytest.warns(UserWarning, match='imaginary part'):
        getattr(simulator, method)(operations[method], {'a': 0.5 + 0.25j})
    reference = _simulator()
    reference.apply_gate(RX(0.5).on(0))
    np.testing.assert_allclose(simulator.get_qs(), reference.get_qs(), atol=1e-12)


@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
def test_precision_conversion_preserves_state(backend):
    """
    Feature: Simulator precision conversion.
    Description: Convert an evolved state from double to single precision and evolve the copy.
    Expectation: Conversion preserves amplitudes and the copy owns independent storage.
    """
    simulator = _simulator(backend)
    simulator.apply_circuit(Circuit().ry(0.4, 0).ry(0.3, 1))
    original = simulator.get_qs()
    converted = simulator.astype(mq.complex64)
    assert converted.dtype == mq.complex64
    np.testing.assert_allclose(converted.get_qs(), original, atol=1e-6)
    converted.apply_circuit(Circuit().x(0))
    np.testing.assert_array_equal(simulator.get_qs(), original)
    assert not np.allclose(converted.get_qs(), original)


@pytest.mark.parametrize(
    'circuit,inputs,match',
    [
        pytest.param(
            Circuit().ry('a', 0).as_encoder(), (np.array([[0.1]]), np.array([0.2])), 'Need one input', id='encoder-only'
        ),
        pytest.param(
            Circuit().ry('a', 0).as_encoder().rz('b', 0),
            (np.array([[0.1]]),),
            'Need two inputs',
            id='encoder-and-ansatz',
        ),
    ],
)
def test_gradient_input_arity(circuit, inputs, match):
    """
    Feature: Batched gradient input roles.
    Description: Call a gradient operator with an incorrect number of encoder and ansatz inputs.
    Expectation: Arity errors identify the expected input count.
    """
    gradient = _simulator().get_expectation_with_grad(_HAMILTONIAN, circuit)
    with pytest.raises(ValueError, match=match):
        gradient(*inputs)
