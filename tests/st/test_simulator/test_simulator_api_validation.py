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
"""Public API validation of the ``Simulator`` wrapper.

The wrapper in ``mindquantum/simulator/mqsim.py`` is the layer that is expected
to reject invalid user input before it reaches the C++ kernels.  These tests
cover that contract: wrong shapes, wrong parameter containers, unsupported
combinations and the diagnostic helpers of the wrapper itself.
"""

import pickle

import numpy as np
import pytest

import mindquantum as mq
from mindquantum import Circuit
from mindquantum.core.gates import RX, AmplitudeDampingChannel, DepolarizingChannel
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator, fidelity

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


def _simulator(backend='mqvector', n_qubits=2, dtype=mq.complex128):
    return Simulator(backend, n_qubits, dtype=dtype, seed=9)


def test_simulator_string_representation():
    """
    Feature: Diagnostic string of the simulator.

    Description: Format state vector simulators with few and many qubits and a density matrix simulator.

    Expectation: The state is shown for ``mqvector`` only and ``repr`` mirrors ``str``.
    """
    small = _simulator('mqvector', 1)
    assert 'mqvector simulator with 1 qubit' in str(small)
    assert 'Current quantum state' in str(small)
    assert '\u27e9' in str(small)

    large = _simulator('mqvector', 4)
    assert '4 qubits' in str(large)
    assert 'Current quantum state' in str(large)

    density = _simulator('mqmatrix', 2)
    assert 'mqmatrix simulator' in str(density)
    assert 'Current quantum state' not in str(density)
    assert repr(small) == str(small)


@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
def test_simulator_pickle_roundtrip(backend):
    """
    Feature: Serialization of the simulator backend.

    Description: Pickle and unpickle a simulator, then evolve the restored object.

    Expectation: Backend description survives the round trip; the native state itself is not serialized.
    """
    simulator = _simulator(backend)
    restored = pickle.loads(pickle.dumps(simulator))
    assert restored.backend.name == backend
    assert restored.n_qubits == simulator.n_qubits
    assert restored.dtype == simulator.dtype
    restored.apply_circuit(Circuit().h(0))
    assert np.isfinite(restored.get_qs()).all()


def test_simulator_astype():
    """
    Feature: Precision conversion of the simulator.

    Description: Convert a double precision simulator to single precision and try an invalid type.

    Expectation: The data type is converted and non-complex types are rejected.
    """
    simulator = _simulator('mqvector', 2)
    converted = simulator.astype(mq.complex64)
    assert converted.dtype == mq.complex64
    np.testing.assert_allclose(converted.get_qs(), simulator.get_qs(), atol=1e-6)
    with pytest.raises(ValueError, match='dtype only support'):
        simulator.astype(np.float64)


@pytest.mark.parametrize('backend', ['mqvector', 'mqmatrix'])
def test_get_qs_ket_format(backend):
    """
    Feature: Ket representation of the quantum state.

    Description: Read the state in Dirac notation as a string for both backend families.

    Expectation: A readable ket string is returned and a non-boolean flag is rejected.
    """
    simulator = _simulator(backend)
    simulator.apply_circuit(Circuit().h(0))
    text = simulator.get_qs(ket=True)
    assert isinstance(text, str)
    assert '0' in text and '1' in text
    with pytest.raises(TypeError, match='ket requires a bool'):
        simulator.get_qs(ket='yes')


def test_density_matrix_ket_marks_mixed_states():
    """
    Feature: Ket representation of a mixed state.

    Description: Convert a depolarized density matrix to its ket representation.

    Expectation: The output is flagged as a mixed state.
    """
    simulator = _simulator('mqmatrix', 1)
    simulator.apply_gate(DepolarizingChannel(0.2).on(0))
    assert 'mixed state' in simulator.get_qs(ket=True)


def test_sampling_validation():
    """
    Feature: Validation of ``Simulator.sampling``.

    Description: Sample circuits without measurement, with invalid shots, missing parameter values and too many qubits.

    Expectation: Every invalid request raises the documented exception.
    """
    simulator = _simulator('mqvector')
    with pytest.raises(ValueError, match='at least one measurement'):
        simulator.sampling(Circuit().h(0))
    with pytest.raises(ValueError, match='not less than 1'):
        simulator.sampling(Circuit().measure('m', 0), shots=0)
    with pytest.raises(TypeError, match='requires an int'):
        simulator.sampling(Circuit().measure('m', 0), shots=1.5)
    with pytest.raises(ValueError, match='need a ParameterResolver'):
        simulator.sampling(Circuit().rx('a', 0).measure('m', 0))
    with pytest.raises(TypeError, match='requires a dict or a ParameterResolver'):
        simulator.sampling(Circuit().rx('a', 0).measure('m', 0), pr=1.5)
    with pytest.raises(ValueError, match='more than simulator qubits'):
        _simulator('mqvector', 1).sampling(Circuit().measure('m', 1))


def test_set_qs_shape_validation():
    """
    Feature: Shape validation of ``Simulator.set_qs``.

    Description: Provide a non power of two vector, a vector of the wrong size and a 3D array.

    Expectation: A ``ValueError`` describes the shape mismatch.
    """
    with pytest.raises(ValueError, match='not power of 2'):
        _simulator('mqvector', 3).set_qs(np.ones(7))
    with pytest.raises(ValueError, match='does not match with simulation qubits'):
        _simulator('mqvector', 3).set_qs(np.ones(4))
    with pytest.raises(ValueError, match='1D vector or 2D density matrix'):
        _simulator('mqmatrix', 1).set_qs(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match='Wrong quantum state'):
        _simulator('mqvector', 1).set_qs(np.zeros(2))


def test_apply_circuit_validation():
    """
    Feature: Validation of ``Simulator.apply_circuit``.

    Description: Apply an oversized circuit, a parameterized circuit without parameters and with wrong names.

    Expectation: Every invalid request raises the documented exception.
    """
    with pytest.raises(ValueError, match='more than simulator qubits'):
        _simulator('mqvector', 1).apply_circuit(Circuit().h(1))
    with pytest.raises(ValueError, match='needs a parameter_resolver'):
        _simulator('mqvector').apply_circuit(Circuit().rx('a', 0))
    with pytest.raises(ValueError, match='not in given parameter resolver'):
        _simulator('mqvector').apply_circuit(Circuit().rx('a', 0), {'b': 1.0})


def test_apply_gate_validation():
    """
    Feature: Validation of ``Simulator.apply_gate``.

    Description: Apply a gate above the simulator width and a parameterized gate without a resolver.

    Expectation: Every invalid request raises the documented exception.
    """
    with pytest.raises(ValueError, match='higher than simulator qubits'):
        _simulator('mqvector', 1).apply_gate(RX(0.1).on(2))
    with pytest.raises(ValueError, match='needs a parameter_resolver'):
        _simulator('mqvector').apply_gate(RX('a').on(0))


def test_apply_gate_ignores_complex_parameters_with_warning():
    """
    Feature: Complex parameter handling of the state vector simulator.

    Description: Apply a rotation whose parameter carries a non-zero imaginary part.

    Expectation: A ``UserWarning`` announces that the imaginary part is dropped, and the gate is applied.
    """
    simulator = _simulator('mqvector')
    with pytest.warns(UserWarning, match='imaginary part will be ignored'):
        simulator.apply_gate(RX('a').on(0), {'a': 1.0 + 2.0j})
    reference = _simulator('mqvector')
    reference.apply_gate(RX(1.0).on(0))
    np.testing.assert_allclose(simulator.get_qs(), reference.get_qs(), atol=1e-12)


def test_get_expectation_validation():
    """
    Feature: Validation of ``Simulator.get_expectation``.

    Description: Pass a non-Hamiltonian, an oversized Hamiltonian, a precision mismatch and inconsistent simulators.

    Expectation: Every invalid request raises the documented exception.
    """
    simulator = _simulator('mqvector')
    hamiltonian = Hamiltonian(QubitOperator('Z0'))
    with pytest.raises(TypeError, match='requires a Hamiltonian'):
        simulator.get_expectation(QubitOperator('Z0'))
    with pytest.raises(ValueError, match='bigger than simulator qubits'):
        _simulator('mqvector', 1).get_expectation(Hamiltonian(QubitOperator('Z1'), mq.complex128))
    with pytest.raises(TypeError, match='Data type of mqvector simulator'):
        simulator.get_expectation(Hamiltonian(QubitOperator('Z0'), mq.complex64))
    with pytest.raises(TypeError, match='different with simulator_left'):
        simulator.get_expectation(hamiltonian, simulator_left=_simulator('mqvector', dtype=mq.complex64))
    with pytest.raises(ValueError, match='same type of this simulator'):
        simulator.get_expectation(hamiltonian, simulator_left=_simulator('mqmatrix'))
    with pytest.raises(ValueError, match='Require a 2 simulator'):
        simulator.get_expectation(hamiltonian, simulator_left=_simulator('mqvector', 3))


def test_get_expectation_non_hermitian_density_matrix():
    """
    Feature: Non-Hermitian expectation on a density matrix.

    Description: Request a bra-ket expectation value with the mqmatrix backend.

    Expectation: A ``NotImplementedError`` explains the missing support.
    """
    simulator = _simulator('mqmatrix')
    with pytest.raises(NotImplementedError, match='Non hermitian case'):
        simulator.get_expectation(Hamiltonian(QubitOperator('Z0')), Circuit().h(0), Circuit().h(0))


def test_get_expectation_grad_validation():
    """
    Feature: Validation of ``Simulator.get_expectation_with_grad``.

    Description: Pass a wrong Hamiltonian container, forbidden circuit combinations and clashing parameter roles.

    Expectation: Every invalid request raises the documented exception.
    """
    simulator = _simulator('mqvector')
    hamiltonian = Hamiltonian(QubitOperator('Z0'))
    with pytest.raises(TypeError, match='requires a Hamiltonian or a list'):
        simulator.get_expectation_with_grad(1, Circuit().h(0))
    with pytest.raises(ValueError, match="doesn't support circ_left"):
        _simulator('mqmatrix').get_expectation_with_grad(hamiltonian, Circuit().h(0), Circuit().h(0))
    with pytest.raises(ValueError, match='not support parameter-shift rule'):
        _simulator('mqmatrix').get_expectation_with_grad(hamiltonian, Circuit().h(0), pr_shift=True)
    with pytest.raises(RuntimeError, match='already set to encoder parameter'):
        simulator.get_expectation_with_grad(hamiltonian, Circuit().rx('a', 0).as_encoder().rx('a', 1))
    with pytest.raises(ValueError, match='cannot have measure gate'):
        simulator.get_expectation_with_grad(hamiltonian, Circuit().measure('m', 0))


def test_partial_trace_validation():
    """
    Feature: Validation of ``Simulator.get_partial_trace``.

    Description: Trace an invalid qubit list on a density matrix simulator and call it on a state vector simulator.

    Expectation: The repeated index, out of range qubit and unsupported backend are rejected.
    """
    density = _simulator('mqmatrix')
    with pytest.raises(ValueError, match='cannot be repeated'):
        density.get_partial_trace([0, 0])
    with pytest.raises(ValueError, match='higher than simulator qubits'):
        density.get_partial_trace(2)
    with pytest.raises(TypeError, match='requires a'):
        density.get_partial_trace(1.5)
    with pytest.raises(ValueError, match='not support partial trace'):
        _simulator('mqvector').get_partial_trace(0)


def test_reduced_density_matrix_validation():
    """
    Feature: Validation of ``Simulator.get_reduced_density_matrix``.

    Description: Request a repeated index, all indices and an out-of-range index.

    Expectation: Every invalid request raises a ``ValueError``.
    """
    density = _simulator('mqmatrix')
    with pytest.raises(ValueError, match='repeated indices'):
        density.get_reduced_density_matrix([0, 0])
    with pytest.raises(ValueError, match='less than total number of qubits'):
        density.get_reduced_density_matrix([0, 1])
    with pytest.raises(ValueError, match='out of range'):
        density.get_reduced_density_matrix(3)


def test_entropy_and_purity_of_mixed_state():
    """
    Feature: Entropy and purity of a density matrix.

    Description: Depolarize one qubit and inspect the two scalar diagnostics.

    Expectation: The purity drops below one and the entropy becomes positive.
    """
    simulator = _simulator('mqmatrix', 1)
    simulator.apply_gate(DepolarizingChannel(0.2).on(0))
    assert simulator.purity() == pytest.approx(0.82)
    assert simulator.entropy() == pytest.approx(0.3250829733914482)
    assert _simulator('mqvector', 1).purity() == 1.0
    assert _simulator('mqvector', 1).entropy() == 0


def test_get_pure_state_vector():
    """
    Feature: Extraction of a pure state vector.

    Description: Read the vector of a pure density matrix and of a mixed one.

    Expectation: The pure case returns the vector and the mixed case raises a ``ValueError``.
    """
    simulator = _simulator('mqvector', 2)
    np.testing.assert_allclose(simulator.get_pure_state_vector(), simulator.get_qs())

    density = _simulator('mqmatrix', 2)
    density.apply_circuit(Circuit().h(0))
    np.testing.assert_allclose(
        np.abs(density.get_pure_state_vector()), np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2), atol=1e-12
    )

    mixed = _simulator('mqmatrix', 1)
    mixed.apply_gate(DepolarizingChannel(0.2).on(0))
    with pytest.raises(ValueError, match='Cannot transform mixed density matrix'):
        mixed.get_pure_state_vector()


def test_get_qs_of_qubits():
    """
    Feature: Reduced state of selected qubits.

    Description: Query an integer, a list, the full register and invalid qubit descriptions.

    Expectation: Pure reductions return a vector, mixed reductions return a matrix and invalid input is rejected.
    """
    density = _simulator('mqmatrix', 3)
    density.apply_circuit(Circuit().h(0).x(1))
    vector = density.get_qs_of_qubits(0)
    assert vector.shape == (2,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
    reduced = density.get_qs_of_qubits([0, 1])
    assert reduced.shape == (4,)
    np.testing.assert_allclose(np.sort(np.abs(reduced)), np.array([0.0, 0.0, 1.0, 1.0]) / np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(density.get_qs_of_qubits([0, 1, 2]), density.get_qs())

    mixed = _simulator('mqmatrix', 2)
    mixed.apply_gate(DepolarizingChannel(0.1).on(0))
    assert mixed.get_qs_of_qubits([0]).shape == (2, 2)

    with pytest.raises(TypeError, match='requires an int or a list'):
        density.get_qs_of_qubits(1.5)
    with pytest.raises(TypeError, match='must be integers'):
        density.get_qs_of_qubits([0, 1.5])
    with pytest.raises(ValueError, match='between 0 and 2'):
        density.get_qs_of_qubits([0, 3])


def test_reset_and_copy_are_independent():
    """
    Feature: State lifecycle of the simulator.

    Description: Copy an evolved simulator, reset the original and evolve the copy further.

    Expectation: The copy keeps its state and the reset simulator returns to ``|0...0>``.
    """
    simulator = _simulator('mqvector')
    simulator.apply_circuit(Circuit().h(0))
    copy = simulator.copy()
    simulator.reset()
    np.testing.assert_allclose(simulator.get_qs(), np.array([1, 0, 0, 0], dtype=complex))
    copy.apply_circuit(Circuit().x(1, 0))
    np.testing.assert_allclose(copy.get_qs(), np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2))


def test_apply_circuit_ignores_complex_parameters_with_warning():
    """
    Feature: Complex parameter handling of the state vector simulator.

    Description: Apply a parameterized circuit whose parameter carries an imaginary part.

    Expectation: A ``UserWarning`` announces that the imaginary part is dropped.
    """
    simulator = _simulator('mqvector')
    with pytest.warns(UserWarning, match='imaginary part will be ignored'):
        simulator.apply_circuit(Circuit().rx('a', 0), {'a': 0.5 + 0.25j})
    reference = _simulator('mqvector')
    reference.apply_circuit(Circuit().rx('a', 0), {'a': 0.5})
    np.testing.assert_allclose(simulator.get_qs(), reference.get_qs(), atol=1e-12)


def test_apply_hamiltonian_precision_mismatch():
    """
    Feature: Validation of ``Simulator.apply_hamiltonian``.

    Description: Apply a single precision Hamiltonian on a double precision simulator.

    Expectation: A ``TypeError`` explains the required precision conversion.
    """
    simulator = _simulator('mqvector')
    with pytest.raises(TypeError, match='Data type of mqvector simulator'):
        simulator.apply_hamiltonian(Hamiltonian(QubitOperator('Z0'), mq.complex64))
    simulator.apply_hamiltonian(Hamiltonian(QubitOperator('Z0')))
    assert simulator.get_qs()[0] == pytest.approx(1.0)


def test_astype_rejects_real_precision():
    """
    Feature: Validation of ``Simulator.astype``.

    Description: Request a real floating point precision for the quantum state.

    Expectation: A ``TypeError`` lists the supported complex precisions.
    """
    with pytest.raises(TypeError, match='dtype should be complex type'):
        _simulator('mqvector').astype(mq.float32)


def test_backend_diagnostic_helpers():
    """
    Feature: Backend diagnostic helpers.

    Description: Query the device description and the matrix of a circuit from the backend wrapper.

    Expectation: The description names the backend and the matrix is the transpose of the circuit matrix.
    """
    backend = _simulator('mqvector', 1).backend
    assert backend.device_name() == '1 qubits mqvector simulator.'
    matrix = backend.get_circuit_matrix(Circuit().h(0), mq.ParameterResolver())
    np.testing.assert_allclose(matrix, Circuit().h(0).matrix(), atol=1e-12)


def test_set_qs_density_matrix_validation():
    """
    Feature: Validation of the density matrix branch of ``Simulator.set_qs``.

    Description: Provide a non-Hermitian matrix, a negative population, a zero matrix and a 2D array
        for the state vector backend.

    Expectation: Every invalid state raises a ``ValueError`` with the specific reason.
    """
    with pytest.raises(ValueError, match='must be hermitian'):
        _simulator('mqmatrix', 1).set_qs(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match='cannot be negative'):
        _simulator('mqmatrix', 1).set_qs(np.diag([-0.5, 1.5]).astype(complex))
    with pytest.raises(ValueError, match='Wrong quantum state'):
        _simulator('mqmatrix', 1).set_qs(np.zeros((2, 2), dtype=complex))
    with pytest.raises(ValueError, match='requires a 1-dimensional array'):
        _simulator('mqvector', 2).set_qs(np.eye(4))


def test_gradient_input_preparation_validation():
    """
    Feature: Input preparation of a gradient operator.

    Description: Call an encoder-only and a both-role gradient operator with the wrong number of inputs.

    Expectation: The caller is told how many parameter arrays are expected.
    """
    hamiltonian = Hamiltonian(QubitOperator('Z0'))
    encoder_only = _simulator('mqvector').get_expectation_with_grad(hamiltonian, Circuit().ry('a', 0).as_encoder())
    with pytest.raises(ValueError, match='Need one input'):
        encoder_only(np.array([[0.1]]), np.array([0.2]))

    both = _simulator('mqvector').get_expectation_with_grad(hamiltonian, Circuit().ry('a', 0).as_encoder().rz('b', 0))
    with pytest.raises(ValueError, match='Need two inputs'):
        both(np.array([[0.1]]))


def test_gradient_hamiltonian_precision_mismatch():
    """
    Feature: Validation of ``Simulator.get_expectation_with_grad``.

    Description: Pass a single precision Hamiltonian to a double precision simulator.

    Expectation: A ``TypeError`` names the precision of the simulator and of the Hamiltonian.
    """
    simulator = _simulator('mqvector')
    with pytest.raises(TypeError, match='but 0th hamiltonian is single precision'):
        simulator.get_expectation_with_grad(Hamiltonian(QubitOperator('Z0'), mq.complex64), Circuit().ry('a', 0))


def test_gradient_noise_circuit_rejects_circ_left():
    """
    Feature: Validation of noise circuits in ``get_expectation_with_grad``.

    Description: Combine a noisy circuit with ``circ_left`` and with a noisy left circuit.

    Expectation: A ``ValueError`` explains that the parameter-shift path has no left circuit support.
    """
    noisy = Circuit().ry('a', 0) + AmplitudeDampingChannel(0.1).on(0)
    simulator = _simulator('mqvector')
    with pytest.raises(ValueError, match='noise circuit use parameter shift rule'):
        simulator.get_expectation_with_grad(Hamiltonian(QubitOperator('Z0')), noisy, circ_left=Circuit().ry('b', 0))
    with pytest.raises(ValueError, match='noise circuit use parameter shift rule'):
        simulator.get_expectation_with_grad(
            Hamiltonian(QubitOperator('Z0')), noisy, simulator_left=_simulator('mqvector')
        )
    with pytest.raises(ValueError, match='noise left circuit not support'):
        simulator.get_expectation_with_grad(Hamiltonian(QubitOperator('Z0')), Circuit().ry('a', 0), circ_left=noisy)


def test_gradient_simulator_left_validation():
    """
    Feature: Validation of ``simulator_left`` in ``get_expectation_with_grad``.

    Description: Pass a simulator of a different backend and of a different width as ``simulator_left``.

    Expectation: Each mismatch raises the documented exception.
    """
    hamiltonian = Hamiltonian(QubitOperator('Z0'))
    simulator = _simulator('mqvector')
    with pytest.raises(ValueError, match='same backend as this simulator'):
        simulator.get_expectation_with_grad(hamiltonian, Circuit().ry('a', 0), simulator_left=_simulator('mqmatrix'))
    with pytest.raises(ValueError, match='same n_qubits as this simulator'):
        simulator.get_expectation_with_grad(hamiltonian, Circuit().ry('a', 0), simulator_left=_simulator('mqvector', 3))


def test_gradient_circuit_wider_than_simulator():
    """
    Feature: Validation of ``Simulator.get_expectation_with_grad``.

    Description: Differentiate a circuit that uses more qubits than the simulator.

    Expectation: A ``ValueError`` reports the width mismatch.
    """
    simulator = _simulator('mqvector', 1)
    with pytest.raises(ValueError, match='but circuit has 2 qubits'):
        simulator.get_expectation_with_grad(Hamiltonian(QubitOperator('Z0')), Circuit().ry('a', 1))


def test_stabilizer_expectation_options():
    """
    Feature: Restricted expectation interface of the stabilizer backend.

    Description: Request an expectation with extra options that the stabilizer cannot honour.

    Expectation: A ``ValueError`` lists the supported arguments.
    """
    simulator = Simulator('stabilizer', 2)
    with pytest.raises(ValueError, match='only supports hamiltonian and circ_right'):
        simulator.get_expectation(Hamiltonian(QubitOperator('Z0')), Circuit().h(0), pr=mq.ParameterResolver({'a': 0.1}))


def test_fidelity_accepts_python_lists_and_rejects_square_mismatch():
    """
    Feature: Input conversion of ``fidelity``.

    Description: Pass Python lists and a matrix with a different number of rows and columns.

    Expectation: Lists are converted and the non square matrix is rejected.
    """
    np.testing.assert_allclose(fidelity([1.0, 0.0], [1.0, 0.0]), 1.0)
    with pytest.raises(ValueError, match='row of matrix is not equal to column'):
        fidelity(np.zeros((2, 3)), np.zeros((2, 3)))
