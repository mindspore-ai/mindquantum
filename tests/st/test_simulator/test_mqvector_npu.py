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
"""Basic correctness checks for the experimental mqvector_npu backend."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import CNOT, H, Rxx, Ryy, Rzz, SWAP
from mindquantum.simulator import Simulator, get_supported_simulator
from mindquantum.simulator.available_simulator import MQVECTOR_NPU_SUPPORTED
from mindquantum.utils import random_circuit

pytestmark = pytest.mark.skipif(not MQVECTOR_NPU_SUPPORTED, reason="mqvector_npu is not available in this build.")


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_is_registered_as_complex64_only():
    """Feature: NPU backend registration.
    Description: Inspect the runtime-supported simulator and dtypes.
    Expectation: Only the implemented complex64 dtype is registered.
    """
    supported = get_supported_simulator()
    assert ["mqvector_npu", mq.complex64] in supported
    assert ["mqvector_npu", mq.complex128] not in supported


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    ("gate_name", "circuit"),
    [
        ("X", Circuit().x(0)),
        ("Y", Circuit().y(0)),
        ("Z", Circuit().h(0).z(0)),
        ("H", Circuit().h(0)),
        ("RX", Circuit().rx(0.314159, 0)),
        ("RY", Circuit().ry(-0.271828, 0)),
        ("RZ", Circuit().h(0).rz(0.314159, 0)),
        ("PS", Circuit().h(0).phase_shift(0.271828, 0)),
        ("CNOT", Circuit([CNOT.on(1, 0)])),
        ("CZ", Circuit().x(0).h(1).z(1, 0)),
    ],
)
def test_mqvector_npu_basic_gate_result(gate_name, circuit):
    """Feature: Basic NPU gate execution.
    Description: Apply each supported one- or two-qubit gate to CPU and NPU.
    Expectation: The complex64 state vectors agree within tolerance.
    """
    cpu = Simulator("mqvector", circuit.n_qubits, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", circuit.n_qubits)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert npu.dtype == mq.complex64
    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5), gate_name


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_copy_and_same_precision_astype():
    """Feature: NPU state copy and conversion.
    Description: Copy a state and convert it to the same complex64 precision.
    Expectation: Both results preserve the source state and are independent.
    """
    npu = Simulator("mqvector_npu", 1)
    npu.apply_circuit(Circuit().h(0).rz(0.314159, 0))

    copied = npu.copy()
    converted = npu.astype(mq.complex64, seed=17)

    assert copied.dtype == mq.complex64
    assert converted.dtype == mq.complex64
    assert np.allclose(copied.get_qs(), npu.get_qs(), atol=1e-5, rtol=1e-5)
    assert np.allclose(converted.get_qs(), npu.get_qs(), atol=1e-5, rtol=1e-5)

    copied.apply_circuit(Circuit().x(0))
    assert not np.allclose(copied.get_qs(), npu.get_qs(), atol=1e-5, rtol=1e-5)
    assert np.allclose(converted.get_qs(), npu.get_qs(), atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_rejects_complex128():
    """Feature: NPU precision validation.
    Description: Construct the backend with the unsupported complex128 dtype.
    Expectation: A TypeError explaining the complex64-only restriction is raised.
    """
    with pytest.raises(TypeError, match="complex64"):
        Simulator("mqvector_npu", 1, dtype=mq.complex128)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_rejects_partial_qubit_state():
    """Feature: NPU reduced-state validation.
    Description: Request a state for a proper subset of the simulator qubits.
    Expectation: NotImplementedError clearly reports the unsupported operation.
    """
    with pytest.raises(NotImplementedError, match="reduced state extraction"):
        Simulator("mqvector_npu", 2).get_qs_of_qubits([0])


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    ("name", "circuit", "n_qubits"),
    [
        ("bell", Circuit([H.on(0), CNOT.on(1, 0)]), 2),
        ("ghz", Circuit([H.on(0), CNOT.on(1, 0), CNOT.on(2, 1)]), 3),
    ],
)
def test_mqvector_npu_typical_circuit_matches_cpu(name, circuit, n_qubits):
    """Feature: Typical circuit correctness.
    Description: Execute Bell and GHZ circuits with CPU and NPU backends.
    Expectation: The complex64 state vectors agree within tolerance.
    """
    cpu = Simulator("mqvector", n_qubits, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", n_qubits)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5), name


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    ("name", "gate"),
    [
        ("Rxx", Rxx(0.271828).on([0, 2])),
        ("Ryy", Ryy(-0.314159).on([0, 2])),
        ("Rzz", Rzz(0.577215).on([0, 2])),
        ("SWAP", SWAP.on([0, 2])),
    ],
)
def test_mqvector_npu_two_qubit_gate_matches_cpu(name, gate):
    """Feature: Two-qubit gate correctness.
    Description: Apply non-adjacent Rxx, Ryy, Rzz, and SWAP gates.
    Expectation: NPU results agree with the complex64 CPU reference.
    """
    circuit = Circuit([H.on(0), H.on(1), H.on(2), gate])
    cpu = Simulator("mqvector", 3, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", 3)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5), name


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_swap_tracked_circuit_matches_cpu():
    """Feature: SWAP tracking.
    Description: Execute a circuit with interleaved SWAP and later target remapping.
    Expectation: Final NPU qubit order and state agree with CPU.
    """
    circuit = Circuit()
    circuit.h(0).ry(0.17, 1).rx(-0.31, 2).phase_shift(0.23, 3)
    circuit += SWAP.on([0, 3])
    circuit += Rxx(0.41).on([0, 2])
    circuit.rz(-0.19, 3)
    circuit += SWAP.on([1, 4])
    circuit += Ryy(-0.37).on([3, 4])
    circuit.z(2, 0)
    circuit += SWAP.on([0, 2])
    circuit += Rzz(0.29).on([1, 3])

    cpu = Simulator("mqvector", 5, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", 5)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    ("name", "circuit"),
    [
        ("controlled H", Circuit().x(1).h(0, 1)),
        ("controlled PS", Circuit().x(1).h(0).phase_shift(0.271828, 0, 1)),
        ("controlled RZ", Circuit().x(1).h(0).rz(-0.314159, 0, 1)),
    ],
)
def test_mqvector_npu_controlled_single_qubit_gate_matches_cpu(name, circuit):
    """Feature: Controlled single-qubit gates.
    Description: Execute controlled H, phase-shift, and RZ circuits.
    Expectation: Controlled NPU states agree with the complex64 CPU reference.
    """
    cpu = Simulator("mqvector", 2, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", 2)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5), name


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_random_circuit_matches_cpu():
    """Feature: Public random-circuit correctness.
    Description: Execute a seeded random_circuit workload on CPU and NPU.
    Expectation: The resulting complex64 state vectors agree within tolerance.
    """
    circuit = random_circuit(4, 32, sd_rate=0.5, ctrl_rate=0.2, seed=20260824)
    cpu = Simulator("mqvector", 4, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", 4)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_mqvector_npu_multitile_random_circuit_matches_cpu():
    """Feature: Multi-tile local circuit execution.
    Description: Execute a 20-qubit random circuit across local-fusion tiles.
    Expectation: Uploaded buffers remain valid and NPU agrees with CPU.
    """
    circuit = random_circuit(20, 100, sd_rate=0.5, ctrl_rate=0.2, seed=20260826)
    cpu = Simulator("mqvector", 20, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", 20)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5)


def _make_seeded_random_circuit(n_qubits, depth, seed):
    """Build a deterministic mixed circuit for CPU/NPU state-vector comparison."""
    rng = np.random.default_rng(seed)
    circuit = Circuit()
    gate_names = ("x", "y", "z", "h", "rx", "ry", "rz", "cnot", "cz")
    for layer in range(depth):
        target = int(rng.integers(0, n_qubits))
        gate_name = str(rng.choice(gate_names))
        if gate_name == "cnot":
            control = int(rng.choice([qubit for qubit in range(n_qubits) if qubit != target]))
            circuit.append(CNOT.on(target, control))
        elif gate_name == "cz":
            control = int(rng.choice([qubit for qubit in range(n_qubits) if qubit != target]))
            circuit.z(target, control)
        elif gate_name == "rx":
            circuit.rx(float(rng.uniform(-np.pi, np.pi)), target)
        elif gate_name == "ry":
            circuit.ry(float(rng.uniform(-np.pi, np.pi)), target)
        elif gate_name == "rz":
            circuit.rz(float(rng.uniform(-np.pi, np.pi)), target)
        else:
            getattr(circuit, gate_name)(target)
    return circuit


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize("n_qubits, depth, seed", [(2, 9, 20260822), (4, 17, 20260823), (6, 25, 20260824)])
def test_mqvector_npu_seeded_random_circuit_matches_cpu(n_qubits, depth, seed):
    """Feature: Seeded mixed-circuit correctness.
    Description: Execute deterministic mixed circuits at several sizes and depths.
    Expectation: NPU state vectors agree with the complex64 CPU reference.
    """
    circuit = _make_seeded_random_circuit(n_qubits, depth, seed)
    cpu = Simulator("mqvector", n_qubits, dtype=mq.complex64)
    npu = Simulator("mqvector_npu", n_qubits)

    cpu.apply_circuit(circuit)
    npu.apply_circuit(circuit)

    assert np.allclose(npu.get_qs(), cpu.get_qs(), atol=1e-5, rtol=1e-5)
