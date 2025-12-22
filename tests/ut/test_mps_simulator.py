# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Tests for mqmps simulator backend."""

import numpy as np
import pytest

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, Measure, RX, RZ, X
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator


def _assert_statevector_close_up_to_global_phase(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-8):
    assert a.shape == b.shape
    inner = np.vdot(a, b)
    if inner == 0:
        assert np.allclose(a, b, atol=atol)
        return
    phase = np.exp(-1j * np.angle(inner))
    assert np.allclose(a, b * phase, atol=atol)


def test_mqmps_statevector_matches_mqvector():
    n_qubits = 4
    circ = Circuit()
    circ += H.on(0)
    circ += RX("a").on(1)
    circ += RZ("b").on(3)
    circ += X.on(3, 0)  # non-adjacent controlled-X (routing needed)
    circ += X.on(2, 1)  # adjacent controlled-X

    pr = {"a": 0.25, "b": -0.7}

    sim_vec = Simulator("mqvector", n_qubits, seed=7)
    sim_mps = Simulator("mqmps", n_qubits, seed=7, max_bond_dimension=0)

    sim_vec.apply_circuit(circ, pr=pr)
    sim_mps.apply_circuit(circ, pr=pr)

    _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-8)


def test_mqmps_expectation_matches_mqvector():
    n_qubits = 4
    prep = Circuit().h(0).rx(0.3, 1).rz(-0.2, 3).x(2, 0)

    ham = Hamiltonian(
        QubitOperator("Z0 Z1", 1.0) + QubitOperator("X2", 0.25) + QubitOperator("Y3", -0.125) + QubitOperator("", 0.5)
    )

    sim_vec = Simulator("mqvector", n_qubits, seed=11)
    sim_mps = Simulator("mqmps", n_qubits, seed=11, max_bond_dimension=0)
    sim_vec.apply_circuit(prep)
    sim_mps.apply_circuit(prep)

    e_vec = sim_vec.get_expectation(ham)
    e_mps = sim_mps.get_expectation(ham)
    assert np.allclose(e_vec, e_mps, atol=1e-8)


def test_mqmps_get_expectation_with_circuit_matches_mqvector():
    n_qubits = 4
    circ = Circuit()
    circ += H.on(0)
    circ += RX("a").on(1)
    circ += RZ("b").on(3)
    circ += X.on(3, 0)
    circ += X.on(2, 1)

    ham = Hamiltonian(
        QubitOperator("Z0 Z1", 1.0) + QubitOperator("X2", 0.25) + QubitOperator("Y3", -0.125) + QubitOperator("", 0.5)
    )

    sim_vec = Simulator("mqvector", n_qubits, seed=21)
    sim_mps = Simulator("mqmps", n_qubits, seed=21, max_bond_dimension=0)

    pr_list = [{"a": 0.25, "b": -0.7}, {"a": -0.33, "b": 1.2}, {"a": 0.0, "b": 0.0}]
    for pr in pr_list:
        e_vec = sim_vec.get_expectation(ham, circ, pr=pr)
        e_mps = sim_mps.get_expectation(ham, circ, pr=pr)
        assert np.allclose(e_vec, e_mps, atol=1e-8)


def test_mqmps_sampling_measure_end_deterministic():
    n_qubits = 5
    circ = Circuit()
    circ += X.on(0)
    circ += X.on(3)
    circ += X.on(4)
    circ.measure_all()

    shots = 50
    seed = 123

    sim_vec = Simulator("mqvector", n_qubits, seed=3)
    sim_mps = Simulator("mqmps", n_qubits, seed=3, max_bond_dimension=0)

    res_vec = sim_vec.sampling(circ, shots=shots, seed=seed)
    res_mps = sim_mps.sampling(circ, shots=shots, seed=seed)

    assert res_vec.bit_string_data == res_mps.bit_string_data
    assert sum(res_mps.bit_string_data.values()) == shots
    assert len(res_mps.bit_string_data) == 1


def test_mqmps_rejects_mid_circuit_measurement():
    n_qubits = 2
    circ = Circuit()
    circ += Measure("q0").on(0)
    circ += H.on(1)
    sim_mps = Simulator("mqmps", n_qubits, seed=5)
    with pytest.raises(RuntimeError):
        sim_mps.apply_circuit(circ)
