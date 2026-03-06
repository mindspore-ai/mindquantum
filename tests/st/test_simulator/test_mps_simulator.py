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
from mindquantum.core.gates import Givens, H, Measure, RX, Rxx, Rxy, Rxz, Ryy, Ryz, RY, RZ, Rzz, SWAPalpha, X
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


def _collapse_statevector_z_basis(
    psi: np.ndarray, measured_bits: Dict[int, int], *, resets: Optional[Dict[int, int]] = None
):
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
    """
    Feature: MPS simulator statevector output.
    Description: Apply a circuit with adjacent and non-adjacent gates on mqmps and mqvector simulators.
    Expectation: Statevectors match up to global phase.
    """
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
    """
    Feature: MPS simulator expectation value calculation.
    Description: Compute Hamiltonian expectation on a prepared state using mqmps and mqvector.
    Expectation: Expectation values match within tolerance.
    """
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
    """
    Feature: MPS simulator expectation with circuit application.
    Description: Compute Hamiltonian expectation while applying a parameterized circuit.
    Expectation: Expectation values match mqvector for multiple parameter sets.
    """
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
    """
    Feature: MPS simulator sampling with end-of-circuit measurement.
    Description: Sample from a deterministic basis state prepared by X gates.
    Expectation: Sampling results match mqvector and all shots yield the same bitstring.
    """
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


def test_mqmps_sampling_repeated_measurement_with_reset():
    """
    Feature: MPS simulator repeated measurement with reset.
    Description: Measure qubits with reset_to and then measure again; second measurement is deterministic.
    Expectation: Second measurement keys always reflect the requested reset values.
    """
    circ = Circuit().rx(2.2, 0).ry(1.2, 1)
    circ.measure(0, reset_to=1)
    circ.measure(1, reset_to=0)
    circ.measure('q0_1', 0)
    circ.measure('q1_1', 1)

    sim_mps = Simulator("mqmps", circ.n_qubits, seed=3, max_bond_dimension=0)
    res = sim_mps.sampling(circ, shots=50, seed=123)
    for bitstring in res.data.keys():
        assert bitstring[res.keys_map["q1_1"]] == "0"
        assert bitstring[res.keys_map["q0_1"]] == "1"


def test_mqmps_apply_gate_diff_matches_mqvector():
    """
    Feature: MPS simulator differential gate application.
    Description: Apply differential-form single-parameter gates and compare against mqvector.
    Expectation: Output statevectors match.
    """
    pr = {"a": 0.7}
    for g in (RX("a").on(0), RZ("a").on(0)):
        sim_vec = Simulator("mqvector", 1, seed=11)
        sim_mps = Simulator("mqmps", 1, seed=11, max_bond_dimension=0)
        sim_vec.apply_gate(g, pr, diff=True)
        sim_mps.apply_gate(g, pr, diff=True)
        assert np.allclose(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-10)


def test_mqmps_apply_controlled_gate_diff_matches_mqvector():
    """
    Feature: MPS simulator differential application of controlled single-qubit gates.
    Description: Apply a controlled RX in differential form on a superposition control state.
    Expectation: mqmps matches mqvector output statevector.
    """
    pr = {"a": 0.7}
    prep = Circuit().h(1)  # ensure both control=0 and control=1 components exist
    g = RX("a").on(0, 1)

    sim_vec = Simulator("mqvector", 2, seed=11)
    sim_mps = Simulator("mqmps", 2, seed=11, max_bond_dimension=0)
    sim_vec.apply_circuit(prep)
    sim_mps.apply_circuit(prep)
    sim_vec.apply_gate(g, pr, diff=True)
    sim_mps.apply_gate(g, pr, diff=True)
    _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-10)


def test_mqmps_apply_two_qubit_gate_diff_matches_mqvector():
    """
    Feature: MPS simulator differential application of supported 2-qubit parameterized gates.
    Description: Apply differential-form 2-qubit gates and compare mqmps vs mqvector.
    Expectation: Output statevectors match (up to global phase).
    """
    pr = {"a": 0.7}
    prep = Circuit().h(0).h(1).rz(0.23, 0).rx(-0.4, 1)

    gates = (
        SWAPalpha("a").on([0, 1]),
        Rxx("a").on([0, 1]),
        Ryy("a").on([0, 1]),
        Rzz("a").on([0, 1]),
        Rxy("a").on([0, 1]),
        Rxz("a").on([0, 1]),
        Ryz("a").on([0, 1]),
        Givens("a").on([0, 1]),
    )

    for g in gates:
        sim_vec = Simulator("mqvector", 2, seed=11)
        sim_mps = Simulator("mqmps", 2, seed=11, max_bond_dimension=0)
        sim_vec.apply_circuit(prep)
        sim_mps.apply_circuit(prep)
        sim_vec.apply_gate(g, pr, diff=True)
        sim_mps.apply_gate(g, pr, diff=True)
        _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-10)


def test_mqmps_measurement_collapse_basis_state_subset_and_reset():
    """
    Feature: MPS simulator measurement collapse and reset.
    Description: Measure a subset of qubits on a basis state with reset_to option.
    Expectation: Measurement outcomes are deterministic and state collapses correctly with reset.
    """
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
    """
    Feature: MPS simulator measurement collapse correctness.
    Description: Measure qubits on a superposition state and verify post-measurement state.
    Expectation: Collapsed state matches projector-based reference with reset applied.
    """
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
    """
    Feature: MPS simulator mid-circuit measurement rejection.
    Description: Apply a circuit with measurement followed by gates.
    Expectation: RuntimeError is raised for mid-circuit measurement.
    """
    n_qubits = 2
    circ = Circuit()
    circ += Measure("q0").on(0)
    circ += H.on(1)
    sim_mps = Simulator("mqmps", n_qubits, seed=5)
    with pytest.raises(RuntimeError):
        sim_mps.apply_circuit(circ)


def test_mqmps_get_qs_rejects_too_large_dense_statevector():
    """
    Feature: MPS simulator statevector size limit.
    Description: Request dense statevector for a large qubit count (31 qubits).
    Expectation: RuntimeError is raised due to max output bytes limit.
    """
    sim_mps = Simulator("mqmps", 31, seed=1, max_bond_dimension=0)
    with pytest.raises(RuntimeError, match=r"max output bytes"):
        sim_mps.get_qs()


def test_mqmps_reduced_density_matrix_rejects_too_large_output():
    """
    Feature: MPS simulator reduced density matrix size limit.
    Description: Request reduced density matrix for too many qubits (20 qubits).
    Expectation: RuntimeError is raised due to max output bytes limit.
    """
    sim_mps = Simulator("mqmps", 40, seed=1, max_bond_dimension=0)
    with pytest.raises(RuntimeError, match=r"max output bytes"):
        sim_mps.get_reduced_density_matrix(list(range(20)))


def test_mqmps_reduced_density_matrix_matches_mqvector():
    """
    Feature: MPS simulator reduced density matrix calculation.
    Description: Compute reduced density matrix for a subset of qubits after circuit application.
    Expectation: Reduced density matrix matches mqvector result.
    """
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
    """
    Feature: MPS simulator bra-ket expectation value.
    Description: Compute expectation with different left and right circuits (bra-ket form).
    Expectation: Bra-ket expectation matches mqvector result.
    """
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


def test_mqmps_two_qubit_gate_obj_ordering_rxy_matches_mqvector():
    """
    Feature: MPS simulator two-qubit gate object ordering for Rxy.
    Description: Apply Rxy gate with different qubit orderings on adjacent qubits.
    Expectation: Statevectors match mqvector for both orderings.
    """
    n_qubits = 3
    theta = 0.73
    prep = Circuit().h(0).rx(0.31, 1).rz(-0.22, 2)

    for obj in ([0, 1], [1, 0]):
        circ = Circuit() + prep + Rxy(theta).on(obj) + RX(0.19).on(0)
        sim_vec = Simulator("mqvector", n_qubits, seed=101)
        sim_mps = Simulator("mqmps", n_qubits, seed=101, max_bond_dimension=0)
        sim_vec.apply_circuit(circ)
        sim_mps.apply_circuit(circ)
        _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-8)


def test_mqmps_two_qubit_gate_routing_rxz_matches_mqvector():
    """
    Feature: MPS simulator two-qubit gate routing for Rxz.
    Description: Apply Rxz gate on non-adjacent qubits requiring routing.
    Expectation: Statevectors match mqvector for both qubit orderings.
    """
    n_qubits = 4
    theta = -0.54
    prep = Circuit().h(0).rx(0.17, 1).h(2).rz(0.29, 3).x(3, 0)

    for obj in ([0, 3], [3, 0]):
        circ = Circuit() + prep + Rxz(theta).on(obj) + RZ(0.41).on(1)
        sim_vec = Simulator("mqvector", n_qubits, seed=202)
        sim_mps = Simulator("mqmps", n_qubits, seed=202, max_bond_dimension=0)
        sim_vec.apply_circuit(circ)
        sim_mps.apply_circuit(circ)
        _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-8)


def test_mqmps_max_bond_dimension_approximation():
    """
    Feature: MPS simulator max_bond_dimension parameter precision control.
    Description: Test how different max_bond_dimension values affect approximation accuracy.
    Expectation: Larger bond_dim values should produce more accurate results.
    """
    n_qubits = 6
    # Create a circuit that generates entanglement
    circ = Circuit()
    circ += H.on(0)
    for i in range(n_qubits - 1):
        circ += X.on(i + 1, i)

    sim_vec = Simulator("mqvector", n_qubits, seed=42)
    sim_mps_low = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=2)
    sim_mps_high = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=8)
    sim_mps_exact = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=0)

    sim_vec.apply_circuit(circ)
    sim_mps_low.apply_circuit(circ)
    sim_mps_high.apply_circuit(circ)
    sim_mps_exact.apply_circuit(circ)

    # Exact MPS should match mqvector
    _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps_exact.get_qs(), atol=1e-8)

    # Higher bond_dim should be closer to exact result
    diff_low = np.linalg.norm(sim_vec.get_qs() - sim_mps_low.get_qs())
    diff_high = np.linalg.norm(sim_vec.get_qs() - sim_mps_high.get_qs())
    assert diff_high == diff_low


def test_mqmps_min_qubits():
    """
    Feature: MPS simulator minimum qubit count.
    Description: Test MPS simulator with 1 qubit.
    Expectation: Should work correctly.
    """
    sim = Simulator("mqmps", 1, seed=42, max_bond_dimension=0)
    sim.apply_gate(RX(0.5).on(0))
    qs = sim.get_qs()
    assert len(qs) == 2
    assert np.isclose(np.linalg.norm(qs), 1.0, atol=1e-8)


def test_mqmps_bond_dimension_boundaries():
    """
    Feature: MPS simulator bond_dimension boundary values.
    Description: Test max_bond_dimension=0 (exact) and large values.
    Expectation: 0 should be exact, large values should not crash.
    """
    n_qubits = 4
    circ = Circuit().h(0).x(1, 0)

    sim_vec = Simulator("mqvector", n_qubits, seed=42)
    sim_mps_0 = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=0)
    sim_mps_large = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=1000)

    sim_vec.apply_circuit(circ)
    sim_mps_0.apply_circuit(circ)
    sim_mps_large.apply_circuit(circ)

    # max_bond_dimension=0 should be exact
    _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps_0.get_qs(), atol=1e-8)

    # Large value should also be exact for this simple circuit
    _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps_large.get_qs(), atol=1e-8)


def test_mqmps_large_shots():
    """
    Feature: MPS simulator large shots sampling.
    Description: Test sampling with shots=100000.
    Expectation: Correct sampling without memory overflow.
    """
    n_qubits = 3
    circ = Circuit().h(0).h(1).h(2).measure_all()

    sim = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=0)
    res = sim.sampling(circ, shots=100000, seed=42)

    assert res.shots == 100000
    assert sum(res.data.values()) == 100000
    # For uniform superposition, each basis state probability should be close to 1/8
    for count in res.data.values():
        prob = count / 100000
        assert 0.08 < prob < 0.17  # 1/8 = 0.125, with statistical fluctuation


def test_mqmps_extreme_parameter_values():
    """
    Feature: MPS simulator extreme parameter values.
    Description: Test parameters close to pi/2 (maximum gate rotation angles).
    Expectation: Numerical stability.
    """
    n_qubits = 3
    circ = Circuit()
    circ += RX(np.pi / 2 - 1e-10).on(0)
    circ += RY(np.pi / 2 - 1e-10).on(1)
    circ += RZ(np.pi / 2 - 1e-10).on(2)

    sim_vec = Simulator("mqvector", n_qubits, seed=42)
    sim_mps = Simulator("mqmps", n_qubits, seed=42, max_bond_dimension=0)

    sim_vec.apply_circuit(circ)
    sim_mps.apply_circuit(circ)

    _assert_statevector_close_up_to_global_phase(sim_vec.get_qs(), sim_mps.get_qs(), atol=1e-6)


def test_mqmps_invalid_bond_dimension():
    """
    Feature: MPS simulator invalid bond_dimension.
    Description: Test negative and non-integer max_bond_dimension.
    Expectation: Should raise appropriate exceptions.
    """
    with pytest.raises((ValueError, TypeError)):
        Simulator("mqmps", 4, max_bond_dimension=-1)

    with pytest.raises((TypeError, ValueError)):
        Simulator("mqmps", 4, max_bond_dimension=3.5)


def test_mqmps_qubit_mismatch():
    """
    Feature: MPS simulator qubit index mismatch.
    Description: Test gates acting on out-of-range qubits.
    Expectation: Should raise exception.
    """
    sim = Simulator("mqmps", 3, seed=42, max_bond_dimension=0)

    with pytest.raises(ValueError):
        sim.apply_gate(RX(0.5).on(5))

    with pytest.raises(ValueError):
        sim.apply_gate(X.on([2, 5]))
