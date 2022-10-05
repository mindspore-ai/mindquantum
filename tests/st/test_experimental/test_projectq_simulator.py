#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# pylint: disable=no-member,pointless-statement,useless-suppression

"""Test C++ ProjectQ simulator."""

import math
import warnings

import pytest

try:
    from mindquantum.experimental import ops, simulator, symengine
    from mindquantum.experimental.circuit import Circuit
except ImportError:
    pytest.skip("MindQuantum experimental C++ module not present", allow_module_level=True)

pytest.skip("Disabled until new C++ simulator is merged", allow_module_level=True)

_HAS_PROJECTQ = True
try:
    from projectq import ops as pq_ops
    from projectq.backends import Simulator as PQ_Simulator
    from projectq.cengines import MainEngine
except ImportError:
    _HAS_PROJECTQ = False

has_projectq = pytest.mark.skipif(not _HAS_PROJECTQ, reason='ProjectQ is not installed')

with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=PendingDeprecationWarning)
    PQ_SqrtXInverse = pq_ops.get_inverse(pq_ops.SqrtX)  # pylint: disable=invalid-name

# ==============================================================================


def angle_idfn(val):
    """Pretty-print angles."""
    if isinstance(val, symengine.Basic):
        return f'sym({val})'
    return f'num({val})'


# ==============================================================================


def mindquantum_setup(seed, n_qubits):
    """Generate default setup for MindQuantum."""
    circuit = Circuit()
    qubits = []
    for _ in range(n_qubits):
        qubits.append(circuit.create_qubit())

    return (qubits, circuit, simulator.projectq.Simulator(seed))


# ------------------------------------------------------------------------------


def projectq_setup(seed, n_qubits):
    """Generate default setup for ProjectQ."""
    eng = MainEngine(backend=PQ_Simulator(seed), engine_list=[])
    qureg = eng.allocate_qureg(n_qubits)
    return (qureg, eng)


# ==============================================================================


def run_mindquantum(gate, n_qubits, seed):
    """Run MindQuantum circuit."""
    qubits, circuit, mq_sim = mindquantum_setup(seed, n_qubits)

    circuit.apply_operator(gate, qubits)
    mq_sim.run_circuit(circuit)

    return mq_sim.cheat()


# ------------------------------------------------------------------------------


def run_projectq(gate, n_qubits, seed):
    """Run ProjectQ circuit."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=PendingDeprecationWarning)

        qubits, eng = projectq_setup(seed, n_qubits)
        if n_qubits == 2:
            gate | (*qubits,)
        else:
            gate | qubits
        eng.flush()
        qubits_map, state = eng.backend.cheat()

        pq_ops.All(pq_ops.Measure) | qubits  # pylint: disable=expression-not-assigned
        eng.flush()

        return (qubits_map, state)


# ==============================================================================


@pytest.mark.cxx_exp_projectq
@pytest.mark.parametrize(
    "mq_gate, pq_gate",
    [
        (ops.H(), pq_ops.H),
        (ops.X(), pq_ops.X),
        (ops.Y(), pq_ops.Y),
        (ops.Z(), pq_ops.Z),
        (ops.Sx(), pq_ops.SqrtX),
        (ops.Sxdg(), PQ_SqrtXInverse),
        (ops.S(), pq_ops.S),
        (ops.Sdg(), pq_ops.Sdag),
        (ops.T(), pq_ops.T),
        (ops.Tdg(), pq_ops.Tdag),
    ],
    ids=lambda x: f'{str(x)}',
)
@has_projectq
def test_single_qubit_gates(mq_gate, pq_gate):
    """
    Description: Test single qubit gates.
    Expectation: Success
    """
    seed = 98138
    n_qubits = 1

    mq_map, mq_state = run_mindquantum(mq_gate, n_qubits, seed)
    pq_map, pq_state = run_projectq(pq_gate, n_qubits, seed)

    assert mq_map == pq_map
    assert pytest.approx(mq_state) == pq_state


# ------------------------------------------------------------------------------


@pytest.mark.cxx_exp_projectq
@pytest.mark.parametrize(
    "mq_gate, pq_gate",
    [
        (ops.Swap(), pq_ops.Swap),
        (ops.SqrtSwap(), pq_ops.SqrtSwap),
    ],
    ids=lambda x: f'{str(x)}',
)
@has_projectq
def test_two_qubit_gates(mq_gate, pq_gate):
    """
    Description: Test two qubit gates.
    Expectation: Success
    """
    seed = 98138
    n_qubits = 2

    mq_map, mq_state = run_mindquantum(mq_gate, n_qubits, seed)
    pq_map, pq_state = run_projectq(pq_gate, n_qubits, seed)

    assert mq_map == pq_map
    assert pytest.approx(mq_state) == pq_state


# ------------------------------------------------------------------------------


@pytest.mark.cxx_exp_projectq
@pytest.mark.parametrize(
    "angle",
    [
        0,
        0.2,
        2.1,
        4.1,
        2 * math.pi,
        4 * math.pi,
    ],
    ids=angle_idfn,
)
@pytest.mark.parametrize(
    "mq_gate, pq_gate",
    [(ops.Rx, pq_ops.Rx), (ops.Ry, pq_ops.Ry), (ops.Rz, pq_ops.Rz), (ops.P, pq_ops.R), (ops.Ph, pq_ops.Ph)],
    ids=lambda x: f'{str(x.__name__)}',
)
@has_projectq
def test_single_qubit_param_gates(angle, mq_gate, pq_gate):
    """
    Description: Test single qubit single parameter gates.
    Expectation: Success
    """
    seed = 98138
    n_qubits = 1

    mq_map, mq_state = run_mindquantum(mq_gate(angle), n_qubits, seed)
    pq_map, pq_state = run_projectq(pq_gate(angle), n_qubits, seed)

    assert mq_map == pq_map
    assert pytest.approx(mq_state) == pq_state


# ------------------------------------------------------------------------------


@pytest.mark.cxx_exp_projectq
@pytest.mark.parametrize(
    "angle",
    [
        0,
        0.2,
        2.1,
        4.1,
        2 * math.pi,
        4 * math.pi,
    ],
    ids=angle_idfn,
)
@pytest.mark.parametrize(
    "mq_gate, pq_gate",
    [(ops.Rxx, pq_ops.Rxx), (ops.Ryy, pq_ops.Ryy), (ops.Rzz, pq_ops.Rzz)],
    ids=lambda x: f'{str(x.__name__)}',
)
@has_projectq
def test_two_qubit_param_gates(angle, mq_gate, pq_gate):
    """
    Description: Test two qubits single parameter gates.
    Expectation: Success
    """
    seed = 98138
    n_qubits = 2

    mq_angle = angle
    if mq_gate in (ops.Rxx, ops.Ryy):
        mq_angle /= 2
    elif mq_gate is ops.Rzz:
        mq_angle *= -1

    mq_map, mq_state = run_mindquantum(mq_gate(mq_angle), n_qubits, seed)
    pq_map, pq_state = run_projectq(pq_gate(angle), n_qubits, seed)

    assert mq_map == pq_map
    assert pytest.approx(mq_state) == pq_state
