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
"""QPU execution interface.

This module defines :class:`~mindquantum.device.QPU`, a minimal execution
abstraction for running circuits on real quantum hardware (or hardware-like
backends).

Design principles:

- Mirror the user-facing workflow of :class:`~mindquantum.simulator.Simulator`
  as closely as possible.
- Cover only the currently required capability surface: sampling, expectation
  value estimation, and gradient computation.
- The only primitive a hardware provider must implement is
  :meth:`QPU.sampling`; expectation values and gradients are provided by
  default via shot-based estimation.
"""

from __future__ import annotations

import abc
import copy
from typing import Dict, Iterator, List, Sequence, Set, Tuple, Union

import numpy as np

from ..core.circuit import Circuit
from ..core.gates import H, Measure, MeasureResult, S
from ..core.operators import Hamiltonian
from ..core.parameterresolver import ParameterResolver
from ..simulator.utils import GradOpsWrapper
from ..utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_ansatz,
    _check_encoder,
    _check_hamiltonian_qubits_number,
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)


def _as_real_float(value, *, name: str, atol: float = 1e-12) -> float:
    """
    Convert a numeric value to a real float, rejecting non-zero imaginary parts.

    Args:
        value (Union[numbers.Number, numpy.ndarray]): Scalar numeric value to
            convert. Single-element arrays are unwrapped automatically.
        name (str): Descriptive name used in error messages.
        atol (float): Absolute tolerance for the imaginary part.
            Default: ``1e-12``.

    Returns:
        float, the real part of *value*.

    Raises:
        TypeError: If *value* is not a scalar number or is a multi-element array.
        ValueError: If the imaginary part exceeds *atol*.
    """

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(f"{name} should be a scalar number, but got an array with shape {value.shape}.")
        value = value.item()
    try:
        cval = complex(value)
    except TypeError as exc:
        raise TypeError(f"{name} should be a scalar number, but got {type(value)}.") from exc
    if abs(cval.imag) > atol:
        raise ValueError(f"{name} should be real-valued for QPU execution, but got {value!r}.")
    return float(cval.real)


def _validate_real_parameterized_circuit(circuit: Circuit) -> None:
    """
    Validate that all intrinsic gate parameters in *circuit* are real-valued.

    QPU execution requires real parameter coefficients and constants. This
    function inspects every parameterized gate and raises early if any
    :class:`~.core.parameterresolver.ParameterResolver` contains a non-real
    coefficient or constant term.

    Args:
        circuit (Circuit): The circuit to validate.

    Raises:
        TypeError: If a coefficient or constant is not a scalar number.
        ValueError: If a coefficient or constant has a non-zero imaginary part.
    """

    for gate in circuit:
        if not getattr(gate, "parameterized", False):
            continue
        if not hasattr(gate, "get_parameters"):
            continue
        for pr_idx, param_resolver in enumerate(gate.get_parameters()):
            if not isinstance(param_resolver, ParameterResolver):
                continue
            _as_real_float(
                param_resolver.const,
                name=f"Constant term of intrinsic parameter #{pr_idx} in {gate!r}",
            )
            for pname in param_resolver.keys():
                _as_real_float(
                    param_resolver[pname],
                    name=f"Coefficient of '{pname}' in intrinsic parameter #{pr_idx} of {gate!r}",
                )


# Simulator backends validate seed in [0, 2**23]. Use an exclusive upper bound for RNG draws.
_SEED_UPPER_BOUND = 2**23 + 1


def _require_gate_parameter_resolvers(gate) -> List[ParameterResolver]:
    """
    Return mutable parameter resolvers from a parameterized gate, or fail fast.

    Args:
        gate: A parameterized gate that exposes ``get_parameters()``.

    Returns:
        list[:class:`~.core.parameterresolver.ParameterResolver`], the mutable
        parameter resolvers attached to *gate*.

    Raises:
        RuntimeError: If the gate does not provide ``get_parameters()``, if
            the return value is not a list/tuple, or if any element is not a
            :class:`~.core.parameterresolver.ParameterResolver`.
    """

    if not hasattr(gate, "get_parameters"):
        raise RuntimeError(f"Parameterized gate {gate!r} does not provide get_parameters().")
    params = gate.get_parameters()
    if not isinstance(params, (list, tuple)):
        raise RuntimeError(f"get_parameters() of {gate!r} should return a list/tuple, but got {type(params)}.")
    params = list(params)
    for pr_idx, param_resolver in enumerate(params):
        if not isinstance(param_resolver, ParameterResolver):
            raise RuntimeError(
                f"Intrinsic parameter #{pr_idx} in {gate!r} should be ParameterResolver, but got {type(param_resolver)}."
            )
    return params


class _QPUGradEvaluator:
    """
    Internal helper that plans and evaluates shot-based gradients on a QPU.

    For each circuit parameter the evaluator selects either the parameter-shift
    rule (when applicable and ``pr_shift=True``) or central finite differences,
    then exposes a single :meth:`grad_ops` callable compatible with
    :class:`~mindquantum.simulator.GradOpsWrapper`.

    Args:
        qpu (QPU): The QPU instance used for sampling.
        hams (list[:class:`~.core.operators.Hamiltonian`]): Hamiltonians whose
            expectations (and gradients) are evaluated.
        circ_right (Circuit): The variational circuit.
        shots (int): Number of measurement shots per circuit execution.
        seed (int): Base seed for the deterministic seed stream.
        fd_gap (float): Step size for central finite-difference gradients.
        pr_shift (bool): Whether to prefer the parameter-shift rule when
            possible.
    """

    _SHIFT_RULE_STANDARD = (np.pi / 2.0, 0.5)
    _SHIFT_RULE_SWAPALPHA = (0.5, np.pi / 2.0)
    _STANDARD_SHIFT_GATES = {"RX", "RY", "RZ", "Rxx", "Ryy", "Rzz", "Rxy", "Rxz", "Ryz", "GP", "PS", "RPS"}

    def __init__(
        self,
        qpu: "QPU",
        hams: List[Hamiltonian],
        circ_right: Circuit,
        *,
        shots: int,
        seed: int,
        fd_gap: float,
        pr_shift: bool,
    ):
        self.qpu = qpu
        self.hams = hams
        self.circ_right = circ_right
        self.shots = shots
        self.seed = seed
        self.fd_gap = fd_gap
        self.pr_shift = pr_shift

        self.encoder_params_name = list(circ_right.encoder_params_name)
        self.ansatz_params_name = list(circ_right.ansatz_params_name)
        self.all_params = self.encoder_params_name + self.ansatz_params_name
        self.encoder_param_index = {p: i for i, p in enumerate(self.encoder_params_name)}
        self.ansatz_param_index = {p: i for i, p in enumerate(self.ansatz_params_name)}

        self.version = "both"
        if not self.ansatz_params_name:
            self.version = "encoder"
        if not self.encoder_params_name:
            self.version = "ansatz"

        self.param_to_occurrences: Dict[str, List[Tuple[int, int]]] = {p: [] for p in self.all_params}
        self.param_to_active_occurrences: Dict[str, List[Tuple[int, int]]] = {p: [] for p in self.all_params}
        self.param_to_occurrence_gate_indices: Dict[str, Set[int]] = {p: set() for p in self.all_params}
        self._collect_parameter_occurrences()
        self.grad_strategy = self._build_grad_strategy()

    def _collect_parameter_occurrences(self):
        """Populate per-parameter occurrence maps by scanning every parameterized gate."""

        for gate_idx, gate in enumerate(self.circ_right):
            if not getattr(gate, "parameterized", False):
                continue
            params = _require_gate_parameter_resolvers(gate)
            for pr_idx, param_resolver in enumerate(params):
                requires_grad_parameters = set(param_resolver.requires_grad_parameters)
                for p in param_resolver.keys():
                    if p not in self.param_to_occurrences:
                        continue
                    occurrence = (gate_idx, pr_idx)
                    self.param_to_occurrences[p].append(occurrence)
                    self.param_to_occurrence_gate_indices[p].add(gate_idx)
                    if p in requires_grad_parameters:
                        self.param_to_active_occurrences[p].append(occurrence)

    def _parameter_shift_rule_for_occurrence(self, gate, pr_idx: int) -> Union[Tuple[float, float], None]:
        """Return ``(shift, prefactor)`` for one gate parameter, or ``None`` if unsupported."""

        if getattr(gate, "ctrl_qubits", None):
            return None

        gate_name = getattr(gate, "name", None)
        if gate_name in self._STANDARD_SHIFT_GATES:
            return self._SHIFT_RULE_STANDARD
        if gate_name == "U3":
            if pr_idx in (0, 1, 2):
                return self._SHIFT_RULE_STANDARD
            return None
        if gate_name == "SWAPalpha":
            return self._SHIFT_RULE_SWAPALPHA
        return None

    def _build_grad_strategy(self) -> Dict[str, Tuple[str, float, float]]:
        """Build a per-parameter gradient strategy mapping ``{name: (mode, shift, prefactor)}``."""

        grad_strategy = {}
        for p in self.all_params:
            active_occurrences = self.param_to_active_occurrences[p]
            if not active_occurrences:
                grad_strategy[p] = ("zero", 0.0, 0.0)
                continue

            if self.pr_shift and len(active_occurrences) == 1:
                gate_idx, pr_idx = active_occurrences[0]
                gate = self.circ_right[gate_idx]
                shift_rule = self._parameter_shift_rule_for_occurrence(gate, pr_idx)
                if shift_rule is not None:
                    params = _require_gate_parameter_resolvers(gate)
                    if pr_idx < len(params) and p in params[pr_idx]:
                        coeff = _as_real_float(params[pr_idx][p], name=f"Coefficient of '{p}' in {gate!r}")
                        if coeff == 0.0:
                            grad_strategy[p] = ("zero", 0.0, 0.0)
                            continue
                        intrinsic_shift, intrinsic_prefactor = shift_rule
                        grad_strategy[p] = ("shift", intrinsic_shift / coeff, coeff * intrinsic_prefactor)
                        continue

            grad_strategy[p] = ("finite_diff", float(self.fd_gap), 1.0)

        return grad_strategy

    def _build_param_perturbed_circuit(
        self, pr_dict: Dict[str, float], target_param: str, delta_on_active_occurrences: float
    ) -> Circuit:
        """Build a copy of the circuit with *target_param* shifted at grad-enabled occurrences."""

        if not self.param_to_occurrences[target_param]:
            return self.circ_right

        active_occurrences = set(self.param_to_active_occurrences[target_param])
        base_value = pr_dict[target_param]
        perturbed = self.circ_right.copy()

        for gate_idx in sorted(self.param_to_occurrence_gate_indices[target_param]):
            gate = self.circ_right[gate_idx]
            gate_new = copy.deepcopy(gate)
            params_new = _require_gate_parameter_resolvers(gate_new)
            for pr_idx, param_resolver in enumerate(params_new):
                if target_param not in param_resolver:
                    continue
                coeff_target = _as_real_float(
                    param_resolver.pop(target_param),
                    name=f"Coefficient of '{target_param}' in {gate!r}",
                )
                target_value = base_value
                if (gate_idx, pr_idx) in active_occurrences:
                    target_value += delta_on_active_occurrences
                param_resolver += coeff_target * target_value

                # Fail fast if gate internals were not updated in-place.
                params_check = _require_gate_parameter_resolvers(gate_new)
                if target_param in params_check[pr_idx]:
                    raise RuntimeError(
                        f"Failed to perturb parameter '{target_param}' in gate {gate!r}; "
                        "get_parameters() should return mutable resolver references."
                    )

            perturbed[gate_idx] = gate_new
        return perturbed

    def _expectations_for_circuit(
        self, circuit: Circuit, pr_dict: Dict[str, float], seed_stream: Iterator[Union[int, None]]
    ) -> np.ndarray:
        """Evaluate expectation values for all Hamiltonians on a single circuit configuration."""

        return np.array(
            [
                self.qpu.get_expectation(h, circ_right=circuit, pr=pr_dict, shots=self.shots, seed=next(seed_stream))
                for h in self.hams
            ],
            dtype=complex,
        )

    def _normalize_input_data(self, *inputs_):
        """Validate and reshape grad-op inputs into ``(encoder_data, ansatz_data, batch_size)``."""

        inputs = list(inputs_)
        for i, item in enumerate(inputs):
            if isinstance(item, list):
                inputs[i] = np.array(item)

        if self.version == "both" and len(inputs) != 2:
            raise ValueError("Need two inputs!")
        if self.version in ("encoder", "ansatz") and len(inputs) != 1:
            raise ValueError("Need one input!")

        if self.version == "both":
            if len(inputs[0].shape) == 1:
                inputs[0] = np.array([inputs[0]])
            _check_encoder(inputs[0], len(self.encoder_params_name))
            _check_ansatz(inputs[1], len(self.ansatz_params_name))
            return inputs[0], inputs[1], inputs[0].shape[0]

        if self.version == "encoder":
            if len(inputs[0].shape) == 1:
                inputs[0] = np.array([inputs[0]])
            _check_encoder(inputs[0], len(self.encoder_params_name))
            return inputs[0], np.array([]), inputs[0].shape[0]

        _check_ansatz(inputs[0], len(self.ansatz_params_name))
        return None, inputs[0], 1

    def _build_pr_dict(self, encoder_data, ansatz_data, batch_idx: int) -> Dict[str, float]:
        """Build a real-valued parameter dict for one batch sample."""

        pr_dict: Dict[str, float] = {}
        if self.encoder_params_name:
            for i, p in enumerate(self.encoder_params_name):
                pr_dict[p] = _as_real_float(encoder_data[batch_idx, i], name=f"Value of parameter '{p}'")
        if self.ansatz_params_name:
            for i, p in enumerate(self.ansatz_params_name):
                pr_dict[p] = _as_real_float(ansatz_data[i], name=f"Value of parameter '{p}'")
        return pr_dict

    def _gradient_vector_for_param(
        self, param_name: str, pr_dict: Dict[str, float], seed_stream: Iterator[Union[int, None]]
    ) -> np.ndarray:
        """Compute the gradient w.r.t. *param_name* across all Hamiltonians."""

        mode, shift, prefactor = self.grad_strategy[param_name]
        if mode == "zero":
            return np.zeros((len(self.hams),), dtype=complex)

        circ_plus = self._build_param_perturbed_circuit(pr_dict, param_name, shift)
        circ_minus = self._build_param_perturbed_circuit(pr_dict, param_name, -shift)
        exp_plus = self._expectations_for_circuit(circ_plus, pr_dict, seed_stream)
        exp_minus = self._expectations_for_circuit(circ_minus, pr_dict, seed_stream)
        if mode == "shift":
            return prefactor * (exp_plus - exp_minus)
        return (exp_plus - exp_minus) / (2.0 * shift)

    def grad_ops(self, *inputs_):
        """Evaluate expectation values and parameter gradients for a batch of inputs."""

        encoder_data, ansatz_data, batch_size = self._normalize_input_data(*inputs_)
        f = np.zeros((batch_size, len(self.hams)), dtype=complex)
        g_enc = np.zeros((batch_size, len(self.hams), len(self.encoder_params_name)), dtype=complex)
        g_ans = np.zeros((batch_size, len(self.hams), len(self.ansatz_params_name)), dtype=complex)
        seed_stream = self.qpu._sampling_seed_stream(self.seed)

        for b in range(batch_size):
            pr_dict = self._build_pr_dict(encoder_data, ansatz_data, b)
            f[b, :] = self._expectations_for_circuit(self.circ_right, pr_dict, seed_stream)

            for p in self.all_params:
                grad_vec = self._gradient_vector_for_param(p, pr_dict, seed_stream)
                if p in self.encoder_param_index:
                    g_enc[b, :, self.encoder_param_index[p]] = grad_vec
                else:
                    g_ans[b, :, self.ansatz_param_index[p]] = grad_vec

        if self.version == "both":
            return f, g_enc, g_ans
        if self.version == "encoder":
            return f, g_enc
        return f, g_ans


# pylint: disable=too-few-public-methods
class QPU(abc.ABC):
    """
    Base class for QPU execution.

    Users should instantiate a concrete QPU subclass provided by a hardware
    vendor (e.g. ``MyVendorQPU(n_qubits, ...)``). Subclasses only need to
    implement :meth:`sampling`; the base class provides default shot-based
    implementations for expectation values and gradients.

    Args:
        n_qubits (int): Number of qubits available on this QPU.
        default_shots (int): Default number of measurement shots used when
            ``shots`` is not explicitly provided. Default: ``1000``.

    Raises:
        TypeError: If *n_qubits* or *default_shots* is not an integer.
        ValueError: If *n_qubits* or *default_shots* is less than 1.

    Note:
        - Only :class:`~.core.operators.Hamiltonian` constructed from
          :class:`~.core.operators.QubitOperator` is supported in the default
          expectation / gradient implementation (hardware-measurable Pauli
          sums).
        - Non-Hermitian expectation values and the ``circ_left`` /
          ``simulator_left`` overlap form are not supported.
        - Circuits passed to :meth:`get_expectation` or
          :meth:`get_expectation_with_grad` must not contain measurement
          gates; measurement bases are generated internally.
        - Hardware constraints (topology, native gates, calibration-aware
          compilation) are intentionally left to concrete subclasses.
        - Providers can override :meth:`_sampling_batch` to submit batched
          circuit jobs to hardware.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import Measure
        >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
        >>> from mindquantum.device import QPU
        >>> class DummyQPU(QPU):
        ...     def sampling(self, circuit, pr=None, shots=1, seed=None):
        ...         # Concrete subclass implements real hardware sampling here.
        ...         from mindquantum.simulator import Simulator
        ...         sim = Simulator('mqvector', self.n_qubits)
        ...         return sim.sampling(circuit, pr=pr, shots=shots, seed=seed)
        >>> qpu = DummyQPU(n_qubits=2, default_shots=1000)
        >>> qpu
        DummyQPU(n_qubits=2, default_shots=1000)
        >>> ham = Hamiltonian(QubitOperator('Z0'))
        >>> circ = Circuit().ry(1.2, 0)
        >>> isinstance(qpu.get_expectation(ham, circ), complex)
        True
    """

    def __init__(self, n_qubits: int, default_shots: int = 1000):
        _check_int_type("n_qubits", n_qubits)
        _check_value_should_not_less("n_qubits", 1, n_qubits)
        _check_int_type("default_shots", default_shots)
        _check_value_should_not_less("default_shots", 1, default_shots)
        self._n_qubits = int(n_qubits)
        self.default_shots = int(default_shots)

    @property
    def n_qubits(self) -> int:
        """
        Get the number of qubits available on this QPU.

        Returns:
            int, the qubit count.
        """

        return self._n_qubits

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}(n_qubits={self.n_qubits}, default_shots={self.default_shots})"

    @staticmethod
    def _sampling_seed_stream(seed: int = None) -> Iterator[Union[int, None]]:
        """
        Create a deterministic stream of per-call sampling seeds.

        If *seed* is ``None``, every yielded value is ``None`` (non-deterministic).
        Otherwise a NumPy RNG seeded with *seed* produces reproducible integers
        in the range ``[0, 2**23]``.
        """

        if seed is None:
            while True:
                yield None
        _check_int_type("seed", seed)
        _check_value_should_not_less("seed", 0, seed)
        rng = np.random.default_rng(seed)
        while True:
            yield int(rng.integers(0, _SEED_UPPER_BOUND))

    def _sampling_batch(
        self, circuits: Sequence[Circuit], pr: ParameterResolver, shots: int, seed: int = None
    ) -> List[MeasureResult]:
        """
        Sample a batch of circuits sequentially.

        The default implementation calls :meth:`sampling` once per circuit.
        Subclasses may override this method to submit a single batched job to
        the hardware provider for better throughput.

        Args:
            circuits (Sequence[Circuit]): Circuits to sample.
            pr (ParameterResolver): Parameter values shared across all circuits.
            shots (int): Number of shots per circuit.
            seed (int): Optional base seed for reproducibility. Default: ``None``.

        Returns:
            list[MeasureResult], one result per circuit in the same order.
        """

        seed_stream = self._sampling_seed_stream(seed)
        return [self.sampling(circuit, pr=pr, shots=shots, seed=next(seed_stream)) for circuit in circuits]

    @abc.abstractmethod
    def sampling(
        self,
        circuit: Circuit,
        pr: Union[Dict, ParameterResolver] = None,
        shots: int = 1,
        seed: int = None,
    ) -> MeasureResult:
        """
        Execute *circuit* on this QPU and return sampling results.

        This is the only method a hardware backend **must** implement.
        All other public methods (expectation, gradients) are built on top of
        this primitive.

        Args:
            circuit (Circuit): Circuit containing at least one measurement gate.
            pr (Union[dict, ParameterResolver]): Parameter values for a
                parameterized circuit. Default: ``None``.
            shots (int): Number of measurement shots. Default: ``1``.
            seed (int): Optional seed for reproducibility (may be ignored by
                real hardware). Default: ``None``.

        Returns:
            MeasureResult, the sampling results.
        """

    # pylint: disable=too-many-arguments
    def get_expectation(
        self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None, *, shots=None, seed=None
    ):
        r"""
        Estimate the expectation value of a Hamiltonian by sampling.

        This method mirrors the user API of
        :meth:`mindquantum.simulator.Simulator.get_expectation`, but the value
        is estimated from measurement shots rather than computed exactly.

        Only the Hermitian Pauli-sum case is supported:

        .. math::

            E = \langle 0 | U^\dagger H U | 0 \rangle

        where :math:`U` is ``circ_right`` and :math:`|0\rangle` denotes the
        implicit all-zero initial state :math:`|0\cdots0\rangle`.

        Args:
            hamiltonian (Hamiltonian): The Hamiltonian to evaluate. Must be
                constructed from :class:`~.core.operators.QubitOperator`.
            circ_right (Circuit): State-preparation circuit :math:`U`.
                Default: ``None`` (empty circuit).
            circ_left: Not supported. Must be ``None``.
            simulator_left: Not supported. Must be ``None``.
            pr (Union[Dict[str, numbers.Number], ParameterResolver, numpy.ndarray, list, numbers.Number]):
                Parameter values for ``circ_right``. Default: ``None``.
            shots (int): Shots per Pauli-term measurement circuit. If ``None``,
                uses :attr:`default_shots`. Default: ``None``.
            seed (int): Optional seed forwarded to :meth:`sampling`.
                Default: ``None``.

        Returns:
            numbers.Number, the estimated expectation value. A complex number
            with zero imaginary part is returned to match the
            :class:`~mindquantum.simulator.Simulator` API.

        Raises:
            TypeError: If *hamiltonian* is not a
                :class:`~.core.operators.Hamiltonian`.
            NotImplementedError: If *circ_left* or *simulator_left* is not
                ``None``, or if the Hamiltonian is not constructed from
                :class:`~.core.operators.QubitOperator`.
            ValueError: If *circ_right* contains measurement gates, or if the
                circuit requires more qubits than the QPU provides.
        """

        from ..core.operators.hamiltonian import HowTo  # pylint: disable=import-outside-toplevel

        _check_input_type("hamiltonian", Hamiltonian, hamiltonian)
        if circ_right is None:
            circ_right = Circuit()
        _check_input_type("circ_right", Circuit, circ_right)

        if circ_left is not None or simulator_left is not None:
            raise NotImplementedError("QPU expectation does not support circ_left/simulator_left yet.")

        if circ_right.has_measure_gate:
            raise ValueError("circuit for expectation evaluation cannot have measure gate")
        _validate_real_parameterized_circuit(circ_right)

        if hamiltonian.how_to != HowTo.ORIGIN:
            raise NotImplementedError(
                "QPU.get_expectation only supports Hamiltonian constructed from QubitOperator (Pauli sums)."
            )
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)

        if circ_right.n_qubits > self.n_qubits:
            raise ValueError(f"QPU has {self.n_qubits} qubits, but circuit has {circ_right.n_qubits} qubits.")

        if shots is None:
            shots = self.default_shots
        _check_int_type("shots", shots)
        _check_value_should_not_less("shots", 1, shots)

        if circ_right.params_name:
            if pr is None:
                raise ValueError("Evaluating a parameterized circuit needs a parameter_resolver.")
            pr = _check_and_generate_pr_type(pr, circ_right.params_name)
            pr = ParameterResolver(
                {name: _as_real_float(pr[name], name=f"Value of parameter '{name}'") for name in circ_right.params_name}
            )
        else:
            pr = ParameterResolver()

        exp_val = 0.0
        pauli_terms: List[Tuple[Tuple[Tuple[int, str], ...], float]] = []
        for term, coeff in hamiltonian.ham_termlist:
            coeff_real = _as_real_float(coeff, name="Hamiltonian coefficient")
            if not term:
                exp_val += coeff_real
                continue
            pauli_terms.append((term, coeff_real))

        if pauli_terms:
            meas_circuits = [self._build_pauli_measurement_circuit(circ_right, term) for term, _ in pauli_terms]
            meas_results = self._sampling_batch(meas_circuits, pr=pr, shots=shots, seed=seed)
            if len(meas_results) != len(meas_circuits):
                raise RuntimeError(
                    f"sampling_batch should return {len(meas_circuits)} results, but got {len(meas_results)}."
                )
            for (_, coeff_real), meas_res in zip(pauli_terms, meas_results):
                exp_val += coeff_real * self._pauli_term_expectation_from_samples(meas_res)

        return complex(exp_val)

    @staticmethod
    def _pauli_term_expectation_from_samples(meas_res: MeasureResult) -> float:
        r"""Compute :math:`\langle P \rangle` from a :class:`MeasureResult` after basis rotation."""

        if not isinstance(meas_res, MeasureResult):
            raise TypeError(f"sampling should return a MeasureResult, but got {type(meas_res)}")
        if meas_res.shots <= 0:
            raise ValueError("MeasureResult has no shots to estimate expectation.")
        samples = np.asarray(meas_res.samples)
        if samples.size == 0:
            raise ValueError("MeasureResult has empty samples.")
        if samples.ndim != 2:
            raise ValueError(f"MeasureResult.samples should be a 2D array, but got shape {samples.shape}.")
        if samples.shape[1] == 0:
            raise ValueError("MeasureResult.samples has no measurement columns.")
        if not np.logical_or(samples == 0, samples == 1).all():
            raise ValueError("MeasureResult.samples should contain binary values 0/1.")
        # Only qubits in the Pauli term are measured, so parity must be computed over
        # the returned measurement columns rather than the full device qubit register.
        parity = np.sum(samples, axis=1) % 2
        eigenvalues = 1 - 2 * parity  # 0 -> +1, 1 -> -1
        return float(np.mean(eigenvalues))

    @staticmethod
    def _build_pauli_measurement_circuit(base_circuit: Circuit, term: Tuple[Tuple[int, str], ...]) -> Circuit:
        """Append basis-rotation gates and measurements for a Pauli term to a copy of *base_circuit*."""

        circ = base_circuit.copy()
        measured_qubits: List[int] = []
        for qubit, op in term:
            measured_qubits.append(qubit)
            if op == "X":
                circ += H.on(qubit)
            elif op == "Y":
                circ += S.on(qubit).hermitian()
                circ += H.on(qubit)
            elif op == "Z":
                pass
            else:
                raise ValueError(f"Unsupported pauli operator {op!r} in Hamiltonian term {term}.")

        for qubit in measured_qubits:
            circ += Measure().on(qubit)
        return circ

    # pylint: disable=too-many-arguments
    def get_expectation_with_grad(
        self,
        hams,
        circ_right,
        circ_left=None,
        simulator_left=None,
        parallel_worker=None,
        pr_shift=False,
        *,
        shots=None,
        seed=None,
        fd_gap: float = 0.001,
    ):
        r"""
        Get a function that returns expectation values and gradients w.r.t. circuit parameters.

        The returned object is a
        :class:`~mindquantum.simulator.GradOpsWrapper`, preserving
        compatibility with MindQuantum's VQA / MindSpore integration.

        For real hardware, gradients are estimated by repeatedly executing
        circuits:

        - When ``pr_shift=True``, a parameter-shift rule is used for
          parameters that appear in exactly one grad-enabled intrinsic
          parameter occurrence and whose gate matches the shift-rule table
          (RX, RY, RZ, Rxx, Ryy, Rzz, Rxy, Rxz, Ryz, RPS, GP, PS, U3,
          and SWAPalpha with its special shift constant). Controlled gates and
          unsupported gates (e.g. Givens, FSim, custom) fall back to finite
          differences.
        - When ``pr_shift=False`` (default), all gradients use central finite
          differences.

        Args:
            hams (Union[:class:`~.core.operators.Hamiltonian`, List[:class:`~.core.operators.Hamiltonian`]]):
                A Hamiltonian or a list of Hamiltonians whose expectations are evaluated.
            circ_right (Circuit): The variational circuit :math:`U`.
            circ_left: Not supported. Must be ``None``.
            simulator_left: Not supported. Must be ``None``.
            parallel_worker (int): Accepted for signature compatibility but
                not used. Default: ``None``.
            pr_shift (bool): Whether to prefer the parameter-shift rule when
                applicable. Default: ``False``.
            shots (int): Shots per circuit execution. If ``None``, uses
                :attr:`default_shots`. Default: ``None``.
            seed (int): Optional base seed for reproducible gradient
                estimation. Default: ``None``.
            fd_gap (float): Step size for central finite-difference gradients.
                Default: ``0.001``.

        Returns:
            GradOpsWrapper, a callable whose signature is
            ``grad_ops(*inputs) -> (f, g_enc[, g_ans])`` where *f* has shape
            ``(batch, n_hams)`` and each gradient array has shape
            ``(batch, n_hams, n_params)``.

        Raises:
            TypeError: If *hams* is not a Hamiltonian or list of Hamiltonians,
                or if *circ_right* is not a Circuit.
            NotImplementedError: If *circ_left* or *simulator_left* is not
                ``None``, or if any Hamiltonian is not constructed from
                :class:`~.core.operators.QubitOperator`.
            ValueError: If *circ_right* contains measurement gates, if the
                circuit exceeds the QPU qubit count, if *fd_gap* is not
                positive, or if encoder and ansatz parameter sets overlap.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.device import QPU
            >>> class DummyQPU(QPU):
            ...     def sampling(self, circuit, pr=None, shots=1, seed=None):
            ...         from mindquantum.simulator import Simulator
            ...         sim = Simulator('mqvector', self.n_qubits)
            ...         return sim.sampling(circuit, pr=pr, shots=shots, seed=seed)
            >>> qpu = DummyQPU(n_qubits=1, default_shots=10000)
            >>> circ = Circuit().ry('a', 0)
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> grad_ops = qpu.get_expectation_with_grad(ham, circ, shots=10000, seed=42)
            >>> f, g = grad_ops(np.array([1.0]))
            >>> f.shape, g.shape
            ((1, 1), (1, 1, 1))
        """

        if circ_left is not None or simulator_left is not None:
            raise NotImplementedError("QPU gradients do not support circ_left/simulator_left yet.")
        _check_input_type("circ_right", Circuit, circ_right)
        if circ_right.has_measure_gate:
            raise ValueError("circuit for variational algorithm cannot have measure gate")
        _validate_real_parameterized_circuit(circ_right)

        if isinstance(hams, Hamiltonian):
            hams = [hams]
        elif not isinstance(hams, list):
            raise TypeError(f"hams requires a Hamiltonian or a list of Hamiltonian, but got {type(hams)}")

        from ..core.operators.hamiltonian import HowTo  # pylint: disable=import-outside-toplevel

        for h_tmp in hams:
            _check_input_type("hams's element", Hamiltonian, h_tmp)
            _check_hamiltonian_qubits_number(h_tmp, self.n_qubits)
            if h_tmp.how_to != HowTo.ORIGIN:
                raise NotImplementedError(
                    "QPU.get_expectation_with_grad only supports Hamiltonian "
                    "constructed from QubitOperator (Pauli sums)."
                )

        if circ_right.n_qubits > self.n_qubits:
            raise ValueError(f"QPU has {self.n_qubits} qubits, but circuit has {circ_right.n_qubits} qubits.")

        if shots is None:
            shots = self.default_shots
        _check_int_type("shots", shots)
        _check_value_should_not_less("shots", 1, shots)

        if not isinstance(fd_gap, (int, float)) or fd_gap <= 0:
            raise ValueError(f"fd_gap should be a positive number, but got {fd_gap!r}.")
        if parallel_worker is not None:
            _check_int_type("parallel_worker", parallel_worker)
            _check_value_should_not_less("parallel_worker", 1, parallel_worker)

        encoder_params_name = list(circ_right.encoder_params_name)
        ansatz_params_name = list(circ_right.ansatz_params_name)
        if set(encoder_params_name) & set(ansatz_params_name):
            raise RuntimeError("Parameter cannot be both encoder and ansatz parameter.")

        grad_evaluator = _QPUGradEvaluator(
            self,
            hams,
            circ_right,
            shots=shots,
            seed=seed,
            fd_gap=float(fd_gap),
            pr_shift=pr_shift,
        )

        grad_wrapper = GradOpsWrapper(
            grad_evaluator.grad_ops,
            hams,
            circ_right,
            circ_right,
            grad_evaluator.encoder_params_name,
            grad_evaluator.ansatz_params_name,
            parallel_worker,
            self,
        )
        grad_str = f'{self.n_qubits} qubit' + ('' if self.n_qubits == 1 else 's')
        grad_str += f' {self.__class__.__name__} VQA Operator'
        grad_wrapper.set_str(grad_str)
        return grad_wrapper
