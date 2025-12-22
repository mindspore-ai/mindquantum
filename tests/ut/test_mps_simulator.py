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

from typing import Dict, Optional

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


def _collapse_statevector_z_basis(psi: np.ndarray, measured_bits: Dict[int, int], *, resets: Optional[Dict[int, int]] = None):
    """Project `psi` onto computational-basis outcomes on the given qubits, then (optionally) apply X resets."""
    out = psi.copy()
    dim = out.size
    for idx in range(dim):
        keep = True
        for q, b in measured_bits.items():
            if ((idx >> q) & 1) != b:
                keep = False
                break
        if not keep:
            out[idx] = 0

    norm = np.linalg.norm(out)
    assert norm != 0
    out /= norm

    if resets:
        for q, reset_to in resets.items():
            if measured_bits[q] == reset_to:
                continue
            mask = 1 << q
            for idx in range(dim):
                if (idx & mask) == 0:
                    out[idx], out[idx | mask] = out[idx | mask], out[idx]
    return out


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


def test_mqmps_measurement_collapse_basis_state_subset_and_reset():
    n_qubits = 4
    circ = Circuit()
    circ += X.on(0)  # prepare |...1> on q0
    circ += X.on(2)  # prepare |..1.> on q2
    circ += X.on(3, 0)  # non-adjacent CNOT (routing), still a basis state
    circ += Measure("m0").on(0)
    circ += Measure("m1").on(1)
    circ += Measure("m3", reset_to=0).on(3)

    sim_mps = Simulator("mqmps", n_qubits, seed=9, max_bond_dimension=0)
    res = sim_mps.apply_circuit(circ)
    assert res is not None

    bitstring = next(iter(res.data))
    b0 = int(bitstring[res.keys_map["m0"]])
    b1 = int(bitstring[res.keys_map["m1"]])
    b3 = int(bitstring[res.keys_map["m3"]])
    assert (b0, b1, b3) == (1, 0, 1)

    expected = np.zeros(1 << n_qubits, dtype=np.complex128)
    expected[(1 << 0) | (1 << 2)] = 1.0  # q3 reset to 0, q2=1, q0=1
    _assert_statevector_close_up_to_global_phase(sim_mps.get_qs(), expected, atol=1e-8)


def test_mqmps_measurement_collapse_matches_projector():
    n_qubits = 4
    prep = Circuit().h(0).x(3, 0).h(2)  # includes non-adjacent 2Q gate (routing)

    sim_mps = Simulator("mqmps", n_qubits, seed=12, max_bond_dimension=0)
    sim_mps.apply_circuit(prep)
    pre_state = sim_mps.get_qs()

    meas = Circuit()
    meas += Measure("m0").on(0)
    meas += Measure("m3", reset_to=1).on(3)
    res = sim_mps.apply_circuit(meas)
    assert res is not None

    bitstring = next(iter(res.data))
    m0 = int(bitstring[res.keys_map["m0"]])
    m3 = int(bitstring[res.keys_map["m3"]])

    expected = _collapse_statevector_z_basis(pre_state, {0: m0, 3: m3}, resets={3: 1})
    _assert_statevector_close_up_to_global_phase(sim_mps.get_qs(), expected, atol=1e-8)


def test_mqmps_rejects_mid_circuit_measurement():
    n_qubits = 2
    circ = Circuit()
    circ += Measure("q0").on(0)
    circ += H.on(1)
    sim_mps = Simulator("mqmps", n_qubits, seed=5)
    with pytest.raises(RuntimeError):
        sim_mps.apply_circuit(circ)


def test_mqmps_get_qs_raises_over_30_qubits():
    sim_mps = Simulator("mqmps", 31, seed=1, max_bond_dimension=0)
    with pytest.raises(RuntimeError, match=r"<=30"):
        sim_mps.get_qs()


def test_mqmps_reduced_density_matrix_matches_mqvector():
    n_qubits = 5
    circ = Circuit()
    circ += H.on(0)
    circ += H.on(2)
    circ += X.on(4, 0)  # non-adjacent controlled-X (routing)
    circ += RX(0.3).on(3)
    circ += X.on(1, 4)  # potentially non-adjacent after routing (stress mapping)

    sim_vec = Simulator("mqvector", n_qubits, seed=17)
    sim_mps = Simulator("mqmps", n_qubits, seed=17, max_bond_dimension=0)
    sim_vec.apply_circuit(circ)
    sim_mps.apply_circuit(circ)

    kept = [4, 0]  # intentionally unsorted; contract should be in logical order
    rho_vec = sim_vec.get_reduced_density_matrix(kept)
    rho_mps = sim_mps.get_reduced_density_matrix(kept)
    assert np.allclose(rho_vec, rho_mps, atol=1e-8)


def test_mqmps_bra_ket_expectation_matches_mqvector():
    n_qubits = 4
    base = Circuit().h(0).rx(0.4, 1).x(3, 0)
    circ_right = Circuit().rz(0.1, 0).x(2, 1)
    circ_left = Circuit().h(3).x(0, 2)

    ham = Hamiltonian(QubitOperator("X0 Y1", 0.3) + QubitOperator("Z2", -0.5) + QubitOperator("Y3", 0.2))

    sim_vec = Simulator("mqvector", n_qubits, seed=31)
    sim_mps = Simulator("mqmps", n_qubits, seed=31, max_bond_dimension=0)
    sim_vec.apply_circuit(base)
    sim_mps.apply_circuit(base)

    e_vec = sim_vec.get_expectation(ham, circ_right, circ_left)
    e_mps = sim_mps.get_expectation(ham, circ_right, circ_left)
    assert np.allclose(e_vec, e_mps, atol=1e-8)
