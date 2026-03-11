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
"""Test QPU interface for hardware execution."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mindquantum.core.circuit import Circuit
from mindquantum.core import gates as G
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.device import QPU
from mindquantum.simulator import Simulator
from mindquantum.utils import random_circuit


class _SamplingOnlyQPU(QPU):
    """A minimal QPU that delegates sampling to an internal simulator."""

    def __init__(self, sim: Simulator, default_shots: int = 1000):
        super().__init__(sim.n_qubits, default_shots=default_shots)
        self._sim = sim

    def sampling(self, circuit, pr=None, shots=1, seed=None):  # pylint: disable=arguments-differ
        return self._sim.sampling(circuit, pr=pr, shots=shots, seed=seed)


class _ExactExpectationQPU(_SamplingOnlyQPU):
    """QPU test double that uses exact simulator expectation inside gradient plumbing."""

    def get_expectation(  # pylint: disable=arguments-differ,too-many-arguments
        self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None, *, shots=None, seed=None
    ):
        _ = (shots, seed)
        return self._sim.get_expectation(
            hamiltonian, circ_right=circ_right, circ_left=circ_left, simulator_left=simulator_left, pr=pr
        )


class _SeedRecordingQPU(_SamplingOnlyQPU):
    """QPU test double that records seeds passed into sampling."""

    def __init__(self, sim: Simulator, default_shots: int = 1000):
        super().__init__(sim, default_shots=default_shots)
        self.seeds = []

    def sampling(self, circuit, pr=None, shots=1, seed=None):  # pylint: disable=arguments-differ
        self.seeds.append(seed)
        return super().sampling(circuit, pr=pr, shots=shots, seed=seed)


def test_chip_get_expectation_pauli_terms():
    """
    Description: Test shot-based expectation estimation for X/Y/Z Pauli terms.
    Expectation: close to simulator exact value.
    """
    circ = Circuit().ry("a", 0)
    pr = {"a": 1.234}

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim, default_shots=5000)

    h_z = Hamiltonian(QubitOperator("Z0"))
    h_x = Hamiltonian(QubitOperator("X0"))

    exact_z = sim.get_expectation(h_z, circ_right=circ, pr=pr)
    exact_x = sim.get_expectation(h_x, circ_right=circ, pr=pr)

    est_z = qpu.get_expectation(h_z, circ_right=circ, pr=pr, shots=5000, seed=42)
    est_x = qpu.get_expectation(h_x, circ_right=circ, pr=pr, shots=5000, seed=42)

    assert np.isclose(est_z.real, exact_z.real, atol=0.06)
    assert np.isclose(est_x.real, exact_x.real, atol=0.06)

    # Y expectation is 0 for a real state prepared by RY.
    h_y = Hamiltonian(QubitOperator("Y0"))
    exact_y = sim.get_expectation(h_y, circ_right=circ, pr=pr)
    est_y = qpu.get_expectation(h_y, circ_right=circ, pr=pr, shots=8000, seed=42)
    assert np.isclose(est_y.real, exact_y.real, atol=0.06)


def test_chip_get_expectation_with_grad_ansatz_only():
    """
    Description: Test expectation-with-grad wrapper on a simple 1-qubit ansatz.
    Expectation: output shapes match simulator convention and gradient is reasonable.
    """
    circ = Circuit().ry("a", 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim, default_shots=6000)

    grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, shots=6000, seed=42)
    f, g = grad_ops(np.array([1.234]))

    assert f.shape == (1, 1)
    assert g.shape == (1, 1, 1)

    # E(a)=cos(a) for <Z> after RY(a)|0>, so dE/da=-sin(a).
    assert np.isclose(f[0, 0].real, np.cos(1.234), atol=0.08)
    assert np.isclose(g[0, 0, 0].real, -np.sin(1.234), atol=0.12)


def test_qpu_get_expectation_with_grad_parameterresolver_linear_ops():
    """
    Description: Validate gradients when gate parameters are ParameterResolver linear expressions.
    Expectation: expectation and gradients match analytic values within shot noise tolerance.
    """

    expr = 0.5 * PR("a") - 1.2 * PR("b") + (0.3 + 0j)  # complex dtype but real-valued expression
    circ = Circuit().ry(expr, 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim, default_shots=12000)

    a_val, b_val = 0.7, -0.2
    theta = 0.5 * a_val - 1.2 * b_val + 0.3
    expected_f = np.cos(theta)
    expected_grad = {"a": -0.5 * np.sin(theta), "b": 1.2 * np.sin(theta)}

    grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, shots=12000, seed=42)
    input_vals = {"a": a_val, "b": b_val}
    pr_arr = np.array([input_vals[p] for p in circ.ansatz_params_name])
    f, g = grad_ops(pr_arr)

    assert f.shape == (1, 1)
    assert g.shape == (1, 1, len(circ.ansatz_params_name))
    assert np.isclose(f[0, 0].real, expected_f, atol=0.06)

    for i, p in enumerate(circ.ansatz_params_name):
        assert np.isclose(g[0, 0, i].real, expected_grad[p], atol=0.10)


def test_qpu_get_expectation_with_grad_all_parameterized_gates():
    """
    Description: Verify QPU gradient evaluation on a circuit that covers all common parameterized gates.
    Expectation: expectation and gradients are close to simulator reference within shot noise tolerance.
    """

    single_parameter_gate = [
        G.RX,
        G.RY,
        G.RZ,
        G.SWAPalpha,
        G.Givens,
        G.GlobalPhase,
        G.PhaseShift,
        G.Rxx,
        G.Rxy,
        G.Rxz,
        G.Ryy,
        G.Ryz,
        G.Rzz,
    ]

    def matrix(alpha):
        ep = 0.5 * (1 + np.exp(1j * np.pi * alpha))
        em = 0.5 * (1 - np.exp(1j * np.pi * alpha))
        return np.array([[ep, em], [em, ep]])

    def diff_matrix(alpha):
        ep = 0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
        em = -0.5 * 1j * np.pi * np.exp(1j * np.pi * alpha)
        return np.array([[ep, em], [em, ep]])

    custom_gate = None
    try:
        custom_gate = G.gene_univ_parameterized_gate("custom", matrix, diff_matrix)
    except ImportError:
        # This optional gate requires numba; keep the rest of the gate-coverage test runnable.
        custom_gate = None

    circ = random_circuit(3, 5, seed=11)
    for i, gate in enumerate(single_parameter_gate):
        gate_ = gate({f"pr0_{i}": 1, f"pr1_{i}": 2})
        circ += gate_.on(list(range(gate_.n_qubits)))
        circ += random_circuit(3, 5, seed=100 + i)
    circ += G.U3("u3_theta", "u3_phi", "u3_lamda").on(0)
    circ += random_circuit(3, 5, seed=201)
    circ += G.U3("u3_theta_1", "u3_phi_1", 1).on(0)
    circ += random_circuit(3, 5, seed=202)
    circ += G.FSim("fsim_theta", "fsim_phi").on([0, 1])
    circ += random_circuit(3, 5, seed=203)
    if custom_gate is not None:
        circ += custom_gate("custom_a").on(0)
    circ = circ.as_ansatz()

    sim = Simulator("mqvector", 3)
    qpu = _SamplingOnlyQPU(sim, default_shots=15000)
    ham = Hamiltonian(QubitOperator("X0") + QubitOperator("Z0"))

    pr = np.random.RandomState(7).rand(len(circ.ansatz_params_name)) * 2 * np.pi
    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, shots=15000, seed=42, fd_gap=0.1)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=False)

    f, g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    assert np.allclose(f.real, ref_f.real, atol=0.12)
    assert np.allclose(g.real, ref_g.real, atol=0.15)


def test_qpu_parameter_shift_controlled_rotation_matches_simulator():
    """
    Description: Controlled parameterized rotations should follow simulator's finite-diff fallback in pr_shift mode.
    Expectation: QPU gradient matches simulator parameter-shift backend for controlled RY.
    """

    circ = Circuit().h(1)
    circ += G.RY("a").on(0, 1)
    circ = circ.as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0") + QubitOperator("Z1") + QubitOperator("X0 X1"))
    pr = np.array([0.37])

    sim = Simulator("mqvector", 2)
    qpu = _ExactExpectationQPU(sim)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, fd_gap=1e-6)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=True)
    qpu_f, qpu_g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    assert np.allclose(qpu_f, ref_f, atol=1e-12)
    assert np.allclose(qpu_g, ref_g, atol=1e-6)


def test_qpu_parameter_shift_keeps_signed_coefficient():
    """
    Description: Parameter-shift on a negative gate coefficient should preserve gradient sign.
    Expectation: QPU gradient equals simulator parameter-shift gradient.
    """

    circ = Circuit().ry({"a": -1.5}, 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    pr = np.array([0.31])

    sim = Simulator("mqvector", 1)
    qpu = _ExactExpectationQPU(sim)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, fd_gap=1e-6)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=True)
    _, qpu_g = qpu_grad_ops(pr)
    _, ref_g = ref_grad_ops(pr)

    assert np.allclose(qpu_g, ref_g, atol=1e-12)


def test_qpu_get_expectation_rejects_measurement_circuit():
    """
    Description: Verify expectation API rejects circuits with measurement gates.
    Expectation: ValueError is raised.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    circ = Circuit().h(0)
    circ += G.Measure().on(0)
    ham = Hamiltonian(QubitOperator("Z0"))
    with pytest.raises(ValueError, match="cannot have measure gate"):
        qpu.get_expectation(ham, circ_right=circ)


def test_qpu_get_expectation_with_grad_fail_fast_for_non_pauli_hamiltonian():
    """
    Description: Validate fail-fast behavior for non-Pauli Hamiltonian in grad API.
    Expectation: NotImplementedError is raised when constructing grad ops.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    circ = Circuit().ry("a", 0).as_ansatz()
    dense = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ham = Hamiltonian(csr_matrix(dense))
    with pytest.raises(NotImplementedError, match="only supports Hamiltonian constructed from QubitOperator"):
        qpu.get_expectation_with_grad(ham, circ)


def test_qpu_get_expectation_rejects_intrinsic_complex_parameter_expression():
    """
    Description: QPU expectation should fail fast on intrinsically complex gate expressions.
    Expectation: ValueError is raised for non-zero imaginary coefficient/constant in gate ParameterResolver.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    ham = Hamiltonian(QubitOperator("Z0"))

    circ_coeff = Circuit().ry((0.5 + 0.1j) * PR("a"), 0)
    with pytest.raises(ValueError, match="should be real-valued for QPU execution"):
        qpu.get_expectation(ham, circ_right=circ_coeff, pr={"a": 0.2})

    circ_const = Circuit().ry(PR({"a": 1.0}, const=0.2j), 0)
    with pytest.raises(ValueError, match="should be real-valued for QPU execution"):
        qpu.get_expectation(ham, circ_right=circ_const, pr={"a": 0.2})


def test_qpu_get_expectation_rejects_complex_runtime_parameter_values():
    """
    Description: QPU expectation should fail fast when runtime parameter values are complex.
    Expectation: ValueError is raised for non-zero imaginary input values.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    ham = Hamiltonian(QubitOperator("Z0"))
    circ = Circuit().ry("a", 0)
    with pytest.raises(ValueError, match="Value of parameter 'a' should be real-valued"):
        qpu.get_expectation(ham, circ_right=circ, pr={"a": 0.2 + 0.1j})


def test_qpu_get_expectation_with_grad_rejects_intrinsic_complex_parameter_expression():
    """
    Description: QPU gradients should fail fast on intrinsically complex gate expressions.
    Expectation: ValueError is raised during grad-ops construction.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    ham = Hamiltonian(QubitOperator("Z0"))

    circ_coeff = Circuit().ry((0.5 + 0.1j) * PR("a"), 0).as_ansatz()
    with pytest.raises(ValueError, match="should be real-valued for QPU execution"):
        qpu.get_expectation_with_grad(ham, circ_coeff, pr_shift=True)

    circ_const = Circuit().ry(PR({"a": 1.0}, const=0.2j), 0).as_ansatz()
    with pytest.raises(ValueError, match="should be real-valued for QPU execution"):
        qpu.get_expectation_with_grad(ham, circ_const, pr_shift=True)


def test_qpu_get_expectation_with_grad_rejects_circ_left():
    """
    Description: Validate unsupported left-overlap API branch.
    Expectation: NotImplementedError is raised.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    circ = Circuit().ry("a", 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    with pytest.raises(NotImplementedError, match="circ_left/simulator_left"):
        qpu.get_expectation_with_grad(ham, circ, circ_left=Circuit())


def test_qpu_sampling_seed_is_decorrelated_between_internal_calls():
    """
    Description: Ensure deterministic seed is expanded into per-call seeds.
    Expectation: Internal sampling calls receive distinct seeds.
    """

    sim = Simulator("mqvector", 2)
    qpu = _SeedRecordingQPU(sim, default_shots=200)
    circ = Circuit().h(0).x(1, 0)
    ham = Hamiltonian(QubitOperator("Z0") + QubitOperator("X0"))
    _ = qpu.get_expectation(ham, circ_right=circ, shots=200, seed=11)
    assert len(qpu.seeds) == 2
    assert len(set(qpu.seeds)) == 2
    assert all(isinstance(seed, int) for seed in qpu.seeds)


def test_qpu_gradient_seed_is_decorrelated_between_evaluations():
    """
    Description: Ensure grad wrapper does not reuse identical seed for all estimations.
    Expectation: Multiple internal sampling calls use multiple seeds.
    """

    sim = Simulator("mqvector", 1)
    qpu = _SeedRecordingQPU(sim, default_shots=200)
    circ = Circuit().u3("a", 0.3, -0.2, 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=False, shots=200, seed=7, fd_gap=0.2)
    grad_ops(np.array([0.7]))
    assert len(qpu.seeds) > 2
    assert len(set(qpu.seeds)) > 1


def test_qpu_grad_respects_no_grad_part_flags():
    """
    Description: Ensure QPU gradients respect no_grad_part flags on gate coefficients.
    Expectation: no-grad parameter gradient is exactly zero and active parameter matches simulator.
    """

    expr = PR({"a": 1.0, "b": 1.0})
    expr.no_grad_part("b")
    circ = Circuit().ry(expr, 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    input_values = {"a": 0.3, "b": -0.1}
    pr = np.array([input_values[p] for p in circ.ansatz_params_name])

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim, default_shots=20000)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, shots=20000, seed=13, fd_gap=0.2)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=False)
    qpu_f, qpu_g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    idx_map = {p: i for i, p in enumerate(circ.ansatz_params_name)}
    assert np.isclose(qpu_f[0, 0].real, ref_f[0, 0].real, atol=0.03)
    assert np.isclose(qpu_g[0, 0, idx_map["a"]].real, ref_g[0, 0, idx_map["a"]].real, atol=0.04)
    assert np.isclose(qpu_g[0, 0, idx_map["b"]].real, 0.0, atol=1e-12)


def test_qpu_grad_respects_mixed_no_grad_on_shared_parameter():
    """
    Description: Verify shared parameter gradient only includes grad-enabled gate occurrences.
    Expectation: QPU gradient matches simulator when one occurrence is no_grad.
    """

    c1 = Circuit().ry("a", 0)
    c1[0].no_grad()
    c2 = Circuit().ry("a", 0)
    circ = (c1 + c2).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    pr = np.array([0.37])

    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim, default_shots=20000)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True, shots=20000, seed=21, fd_gap=0.2)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=False)
    qpu_f, qpu_g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    assert np.isclose(qpu_f[0, 0].real, ref_f[0, 0].real, atol=0.03)
    assert np.isclose(qpu_g[0, 0, 0].real, ref_g[0, 0, 0].real, atol=0.04)


def test_qpu_grad_default_fd_gap_stable_for_large_linear_coefficient():
    """
    Description: Ensure default finite-diff mode remains numerically stable for large linear coefficients.
    Expectation: QPU gradient matches simulator reference without overriding fd_gap.
    """

    circ = Circuit().ry({"a": 10.0}, 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    pr = np.array([0.27])

    sim = Simulator("mqvector", 1)
    qpu = _ExactExpectationQPU(sim)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=False)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=False)
    qpu_f, qpu_g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    assert np.allclose(qpu_f, ref_f, atol=1e-12)
    assert np.allclose(qpu_g, ref_g, atol=1e-4)


def test_qpu_pr_shift_fallback_default_fd_gap_stable_for_large_linear_coefficient():
    """
    Description: Ensure pr_shift fallback finite-diff path is stable for unsupported gates with large coefficients.
    Expectation: QPU gradient is close to simulator parameter-shift backend without overriding fd_gap.
    """

    circ = Circuit().h(0)
    circ += G.Givens({"a": 50.0}).on([0, 1])
    circ = circ.as_ansatz()
    ham = Hamiltonian(QubitOperator("X0") + QubitOperator("Y0 Y1") + QubitOperator("Z1"))
    pr = np.array([0.27])

    sim = Simulator("mqvector", 2)
    qpu = _ExactExpectationQPU(sim)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=True)
    qpu_f, qpu_g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    assert np.allclose(qpu_f, ref_f, atol=1e-12)
    assert np.allclose(qpu_g, ref_g, atol=0.2)


def test_qpu_parameter_shift_rot_pauli_string_matches_reference():
    """
    Description: Verify RotPauliString uses parameter-shift in QPU grad mode.
    Expectation: QPU pr_shift gradient matches simulator finite-diff/adjoin reference.
    """

    circ = Circuit()
    circ += G.RotPauliString("XY", {"a": 1.7}).on([0, 1])
    circ = circ.as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0") + QubitOperator("X1"))
    pr = np.array([0.27])

    sim = Simulator("mqvector", 2)
    qpu = _ExactExpectationQPU(sim)

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=False)
    qpu_f, qpu_g = qpu_grad_ops(pr)
    ref_f, ref_g = ref_grad_ops(pr)

    assert np.allclose(qpu_f, ref_f, atol=1e-12)
    assert np.allclose(qpu_g, ref_g, atol=1e-6)


def test_qpu_grad_perturbed_circuit_preserves_encoder_ansatz_behavior():
    """
    Description: Ensure perturbed circuits preserve encoder/ansatz behavior in mixed-input mode.
    Expectation: QPU outputs match simulator reference in both-mode gradients.
    """

    circ = Circuit().ry("e", 0).as_encoder()
    circ += Circuit().ry("a", 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))

    sim = Simulator("mqvector", 1)
    qpu = _ExactExpectationQPU(sim)

    encoder_batch = np.array([[0.2], [0.7]])
    ansatz_data = np.array([0.3])

    qpu_grad_ops = qpu.get_expectation_with_grad(ham, circ, pr_shift=True)
    ref_grad_ops = sim.get_expectation_with_grad(ham, circ, pr_shift=False)
    qpu_f, qpu_g_enc, qpu_g_ans = qpu_grad_ops(encoder_batch, ansatz_data)
    ref_f, ref_g_enc, ref_g_ans = ref_grad_ops(encoder_batch, ansatz_data)

    assert np.allclose(qpu_f, ref_f, atol=1e-12)
    assert np.allclose(qpu_g_enc, ref_g_enc, atol=1e-6)
    assert np.allclose(qpu_g_ans, ref_g_ans, atol=1e-6)


def test_qpu_min_qubits():
    """
    Feature: QPU minimum qubit count.
    Description: Test QPU with 1 qubit.
    Expectation: Should work correctly.
    """
    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)
    assert qpu.n_qubits == 1

    circ = Circuit().ry("a", 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))
    grad_ops = qpu.get_expectation_with_grad(ham, circ, shots=1000, seed=42)
    f, g = grad_ops(np.array([0.5]))

    assert f.shape == (1, 1)
    assert g.shape == (1, 1, 1)


def test_qpu_shots_boundaries():
    """
    Feature: QPU shots boundary values.
    Description: Test shots=1 and large shots values.
    Expectation: Should handle correctly.
    """
    sim = Simulator("mqvector", 1)
    qpu = _SamplingOnlyQPU(sim)

    circ = Circuit().ry("a", 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))

    # shots=1
    f1, g1 = qpu.get_expectation_with_grad(ham, circ, shots=1, seed=42)(np.array([0.5]))
    assert not np.isnan(f1[0, 0])

    # Large shots
    f2, g2 = qpu.get_expectation_with_grad(ham, circ, shots=100000, seed=42)(np.array([0.5]))
    assert not np.isnan(f2[0, 0])


def test_qpu_fd_gap_boundaries():
    """
    Feature: QPU fd_gap boundary values.
    Description: Test fd_gap close to 0 and large values.
    Expectation: Should handle boundary cases.
    """
    sim = Simulator("mqvector", 1)
    qpu = _ExactExpectationQPU(sim)

    circ = Circuit().ry("a", 0).as_ansatz()
    ham = Hamiltonian(QubitOperator("Z0"))

    # Small fd_gap
    grad_ops_small = qpu.get_expectation_with_grad(ham, circ, fd_gap=1e-6)
    f_small, g_small = grad_ops_small(np.array([0.5]))
    assert not np.isnan(g_small[0, 0, 0])

    # Large fd_gap
    grad_ops_large = qpu.get_expectation_with_grad(ham, circ, fd_gap=1.0)
    f_large, g_large = grad_ops_large(np.array([0.5]))
    assert not np.isnan(g_large[0, 0, 0])
