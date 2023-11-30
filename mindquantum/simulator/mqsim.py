# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Mindquantum simulator."""
from typing import Dict, Iterable, List, Union

import numpy as np
from scipy.linalg import eigvals

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BarrierGate, BasicGate, Measure, MeasureResult
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.dtype import complex128
from mindquantum.dtype.dtype import mq_complex_number_type
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR
from mindquantum.utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_ansatz,
    _check_encoder,
    _check_gate_has_obj,
    _check_hamiltonian_qubits_number,
    _check_input_type,
    _check_int_type,
    _check_mq_type,
    _check_seed,
    _check_value_should_not_less,
)

# This import is required to register some of the C++ types (e.g. ParameterResolver)
from ..utils.string_utils import ket_string
from .backend_base import BackendBase
from .utils import GradOpsWrapper, _thread_balance


# pylint: disable=abstract-method,too-many-arguments
class MQSim(BackendBase):
    """Mindquantum Backend."""

    def __init__(self, name: str, n_qubits: int, seed=None, dtype=complex128, internal=False):
        """Initialize a mindquantum backend."""
        super().__init__(name, n_qubits, seed, dtype)
        if internal:
            self.sim = name
        else:
            if dtype is None:
                dtype = complex128
            _check_mq_type(dtype)
            self.sim = getattr(SUPPORTED_SIMULATOR.c_module(name, dtype), name)(n_qubits, seed)

    def __str__(self):
        """Return a string representation of the object."""
        state = self.get_qs()
        ret = f"{self.name} simulator with {self.n_qubits} qubit{'s' if self.n_qubits > 1 else ''} "
        ret += f"(little endian), dtype: {self.dtype}."
        if self.name.startswith('mqvector'):
            ret += "\nCurrent quantum state:\n"
            if self.n_qubits < 4:
                ret += '\n'.join(ket_string(state))
            else:
                ret += state.__str__()
        return ret

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    def apply_circuit(
        self,
        circuit: Circuit,
        pr: Union[Dict, ParameterResolver] = None,
    ):
        """Apply a quantum circuit."""
        _check_input_type('circuit', Circuit, circuit)
        if self.n_qubits < circuit.n_qubits:
            raise ValueError(f"Circuit has {circuit.n_qubits} qubits, which is more than simulator qubits.")
        if circuit.params_name:
            if pr is None:
                raise ValueError("Applying a parameterized circuit needs a parameter_resolver.")
            pr = _check_and_generate_pr_type(pr, circuit.params_name)
        else:
            pr = ParameterResolver()
        res = self.sim.apply_circuit(circuit.get_cpp_obj(), pr)
        if res:
            out = MeasureResult()
            out.add_measure(circuit.all_measures.keys())
            out.collect_data([[res[i] for i in out.keys_map]])
            return out
        return None

    def apply_gate(
        self,
        gate: BasicGate,
        pr: Union[Dict, ParameterResolver] = None,
        diff: bool = False,
    ):
        """Apply a quantum gate."""
        _check_input_type("gate", BasicGate, gate)
        _check_gate_has_obj(gate)
        if not isinstance(gate, BarrierGate):
            gate_max = max(max(gate.obj_qubits, gate.ctrl_qubits))
            if self.n_qubits <= gate_max:
                raise ValueError(f"qubits of gate {gate} is higher than simulator qubits.")
            if gate.parameterized:
                if pr is None:
                    raise ValueError("apply a parameterized gate needs a parameter_resolver")
                pr = _check_and_generate_pr_type(pr, gate.coeff.params_name)
            else:
                pr = ParameterResolver()
            if isinstance(gate, Measure):
                return self.sim.apply_gate(gate.get_cpp_obj(), pr.get_cpp_obj(), diff)
            self.sim.apply_gate(gate.get_cpp_obj(), pr, diff)
        return None

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """Apply a hamiltonian."""
        if not mq.is_same_precision(self.dtype, hamiltonian.dtype):
            raise TypeError(
                f"Data type of {self.name} simulator is {mq.precision_str(self.dtype)} ({self.dtype}), "
                f"but given hamiltonian is {mq.precision_str(hamiltonian.dtype)} ({hamiltonian.dtype}). "
                f"Please convert given hamiltonian to {mq.precision_str(self.dtype)} "
                f"({mq.to_precision_like(hamiltonian.dtype, self.dtype)})."
            )
        _check_input_type('hamiltonian', Hamiltonian, hamiltonian)
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)
        self.sim.apply_hamiltonian(hamiltonian.get_cpp_obj())

    def astype(self, dtype, seed):
        """Convert simulator to other type."""
        _check_mq_type(dtype)
        if dtype not in mq_complex_number_type:
            raise TypeError(f"dtype should be complex type, available types are {mq_complex_number_type}")
        sim = MQSim(getattr(self.sim, str(dtype).rsplit('.', maxsplit=1)[-1])(seed), self.n_qubits, internal=True)
        sim.name = self.sim.sim_name()
        return sim

    @property
    def dtype(self):
        """Get data type of simulator."""
        return self.sim.dtype()

    def copy(self) -> "BackendBase":
        """Copy a simulator."""
        sim = MQSim(self.name, self.n_qubits, self.seed)
        sim.sim = self.sim.copy()
        return sim

    def device_name(self) -> str:
        """Return the device name."""
        return f"{self.n_qubits} qubits {self.name} simulator."

    def get_circuit_matrix(self, circuit: Circuit, pr: ParameterResolver) -> np.ndarray:
        """Get the matrix of given circuit."""
        return np.array(self.sim.get_circuit_matrix(circuit.get_cpp_obj(), pr)).T

    # pylint: disable=too-many-branches
    def get_expectation(
        self, hamiltonian: Hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None
    ) -> np.ndarray:
        """Get expectation of a hamiltonian."""
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError(f"hamiltonian requires a Hamiltonian, but got {type(hamiltonian)}")
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)
        if not mq.is_same_precision(self.dtype, hamiltonian.dtype):
            raise TypeError(
                f"Data type of {self.name} simulator is {mq.precision_str(self.dtype)} ({self.dtype}), "
                f"but given hamiltonian is {mq.precision_str(hamiltonian.dtype)} ({hamiltonian.dtype}). "
                f"Please convert given hamiltonian to {mq.precision_str(self.dtype)} "
                f"({mq.to_precision_like(hamiltonian.dtype, self.dtype)})."
            )
        hermitian = True
        if circ_right is None:
            circ_right = Circuit()
        if circ_left is None:
            circ_left = circ_right
        else:
            hermitian = False
        if simulator_left is not None:
            hermitian = False
            if self.dtype != simulator_left.dtype:
                raise TypeError(
                    f"The dtype of this simulator ({self.dtype}) is different"
                    f" with simulator_left ({simulator_left.dtype})"
                )
            if self.name != simulator_left.backend.name:
                raise ValueError(
                    f"simulator_left should be the same type of this simulator ({self.name})"
                    f" but get {simulator_left.backend.name}."
                )
            if self.n_qubits != simulator_left.n_qubits:
                raise ValueError(f"Require a {self.n_qubits} simulator, but get {simulator_left.n_qubits}.")
        if pr is None:
            pr = ParameterResolver()
        else:
            pr = ParameterResolver(pr)
        if hermitian:
            return self.sim.get_expectation(hamiltonian.get_cpp_obj(), circ_right.get_cpp_obj(), pr)
        if self.name == 'mqmatrix':
            raise NotImplementedError("Non hermitian case for get_expectation not implement for mqmatrix yet.")
        if simulator_left is None:
            return self.sim.get_expectation(
                hamiltonian.get_cpp_obj(), circ_right.get_cpp_obj(), circ_left.get_cpp_obj(), pr
            )
        return self.sim.get_expectation(
            hamiltonian.get_cpp_obj(), circ_right.get_cpp_obj(), circ_left.get_cpp_obj(), simulator_left.backend.sim, pr
        )

    def get_expectation_with_grad(  # pylint: disable=R0912,R0913,R0914,R0915
        self,
        hams: List[Hamiltonian],
        circ_right: Circuit,
        circ_left: Circuit = None,
        simulator_left: "BackendBase" = None,
        parallel_worker: int = None,
        pr_shift: bool = False,
    ):
        """Get expectation with grad."""
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        elif not isinstance(hams, list):
            raise TypeError(f"hams requires a Hamiltonian or a list of Hamiltonian, but get {type(hams)}")
        for i, ham in enumerate(hams):
            if not mq.is_same_precision(self.dtype, ham.dtype):
                raise TypeError(
                    f"Data type of {self.name} simulator is {mq.precision_str(self.dtype)} ({self.dtype}),"
                    f" but {i}th hamiltonian is {mq.precision_str(ham.dtype)} ({ham.dtype}). "
                    f"Please convert {i}th hamiltonian to {mq.precision_str(self.dtype)} "
                    f"({mq.to_precision_like(ham.dtype, self.dtype)})."
                )
        for h_tmp in hams:
            _check_input_type("hams's element", Hamiltonian, h_tmp)
            _check_hamiltonian_qubits_number(h_tmp, self.n_qubits)
        _check_input_type("circ_right", Circuit, circ_right)
        if circ_right.is_noise_circuit and "mqvector" in self.name:
            if circ_left is not None or simulator_left is not None:
                raise ValueError(
                    "noise circuit use parameter shift rule to get grad, \
                    which not support circ_left and simulator_left."
                )
            pr_shift = True
        if pr_shift and not self.name.startswith('mqvector'):
            raise ValueError(f"{self.name} simulator not support parameter-shift rule.")
        non_hermitian = False
        if circ_left is not None:
            _check_input_type("circ_left", Circuit, circ_left)
            if circ_left.is_noise_circuit:
                raise ValueError("noise left circuit not support yet.")
            non_hermitian = True
        if simulator_left is not None:
            _check_input_type("simulator_left", MQSim, simulator_left)
            if self.name != simulator_left.name:
                raise ValueError(
                    "simulator_left should have the same backend as this simulator, "
                    f"which is {self.name}, but get {simulator_left.name}",
                )
            if self.n_qubits != simulator_left.n_qubits:
                raise ValueError(
                    "simulator_left should have the same n_qubits as this simulator, "
                    f"which is {self.n_qubits}, but get {simulator_left.n_qubits}",
                )
            if self.dtype != simulator_left.dtype:
                raise ValueError(
                    "simulator_left should have the same data type as this simulator "
                    f"(aka {self.dtype}), but get {simulator_left.dtype}.",
                )
            non_hermitian = True
        if non_hermitian and simulator_left is None:
            simulator_left = self
        if circ_left is None:
            circ_left = circ_right
        if circ_left.has_measure_gate or circ_right.has_measure_gate:
            raise ValueError("circuit for variational algorithm cannot have measure gate")
        if parallel_worker is not None:
            _check_int_type("parallel_worker", parallel_worker)

        ansatz_params_name = circ_right.all_ansatz.keys()
        encoder_params_name = circ_right.all_encoder.keys()
        if non_hermitian:
            for i in circ_left.all_ansatz.keys():
                if i not in ansatz_params_name:
                    ansatz_params_name.append(i)
            for i in circ_left.all_encoder.keys():
                if i not in encoder_params_name:
                    encoder_params_name.append(i)
        if set(ansatz_params_name) & set(encoder_params_name):
            raise RuntimeError("Parameter cannot be both encoder and ansatz parameter.")
        version = "both"
        if not ansatz_params_name:
            version = "encoder"
        if not encoder_params_name:
            version = "ansatz"

        circ_n_qubits = max(circ_left.n_qubits, circ_right.n_qubits)
        if self.n_qubits < circ_n_qubits:
            raise ValueError(f"Simulator has {self.n_qubits} qubits, but circuit has {circ_n_qubits} qubits.")

        def grad_ops(*inputs_):
            inputs = list(inputs_)
            for i, item in enumerate(inputs):
                if isinstance(item, list):
                    inputs[i] = np.array(item)
            if version == "both" and len(inputs) != 2:
                raise ValueError("Need two inputs!")
            if version in ("encoder", "ansatz") and len(inputs) != 1:
                raise ValueError("Need one input!")
            if version == "both":
                if len(inputs[0].shape) == 1:
                    inputs[0] = np.array([inputs[0]])
                _check_encoder(inputs[0], len(encoder_params_name))
                _check_ansatz(inputs[1], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = inputs[1]
            if version == "encoder":
                if len(inputs[0].shape) == 1:
                    inputs[0] = np.array([inputs[0]])
                _check_encoder(inputs[0], len(encoder_params_name))
                batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = np.array([])
            if version == "ansatz":
                _check_ansatz(inputs[0], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(1, len(hams), parallel_worker)
                inputs0 = np.array([[]])
                inputs1 = inputs[0]
            if non_hermitian:
                f_g1_g2 = self.sim.get_expectation_with_grad_non_hermitian_multi_multi(
                    [i.get_cpp_obj() for i in hams],
                    [i.get_cpp_obj(hermitian=True) for i in hams],
                    circ_left.get_cpp_obj(),
                    circ_left.get_cpp_obj(hermitian=True),
                    circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True),
                    inputs0,
                    inputs1,
                    encoder_params_name,
                    ansatz_params_name,
                    simulator_left.sim,
                    batch_threads,
                    mea_threads,
                )
            elif circ_right.is_noise_circuit and self.name.startswith('mqmatrix'):
                f_g1_g2 = self.sim.get_expectation_with_noise_grad_multi_multi(
                    [i.get_cpp_obj() for i in hams],
                    circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True),
                    inputs0,
                    inputs1,
                    encoder_params_name,
                    ansatz_params_name,
                    batch_threads,
                    mea_threads,
                )
            elif pr_shift:
                f_g1_g2 = self.sim.get_expectation_with_grad_parameter_shift_multi_multi(
                    [i.get_cpp_obj() for i in hams],
                    circ_right.get_cpp_obj(),
                    inputs0,
                    inputs1,
                    encoder_params_name,
                    ansatz_params_name,
                    batch_threads,
                    mea_threads,
                )
            else:
                f_g1_g2 = self.sim.get_expectation_with_grad_multi_multi(
                    [i.get_cpp_obj() for i in hams],
                    circ_right.get_cpp_obj(),
                    circ_right.get_cpp_obj(hermitian=True),
                    inputs0,
                    inputs1,
                    encoder_params_name,
                    ansatz_params_name,
                    batch_threads,
                    mea_threads,
                )
            res = np.array(f_g1_g2)
            if version == 'both':
                return (
                    res[:, :, 0],
                    res[:, :, 1 : 1 + len(encoder_params_name)],  # noqa:E203
                    res[:, :, 1 + len(encoder_params_name) :],  # noqa:E203
                )  # f, g1, g2
            return res[:, :, 0], res[:, :, 1:]  # f, g

        grad_wrapper = GradOpsWrapper(
            grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker, self
        )
        grad_str = f'{self.n_qubits} qubit' + ('' if self.n_qubits == 1 else 's')
        grad_str += f' {self.name} VQA Operator'
        grad_wrapper.set_str(grad_str)
        return grad_wrapper

    def get_qs(self, ket=False) -> np.ndarray:
        """Get quantum state of mqvector simulator."""
        if not isinstance(ket, bool):
            raise TypeError(f"ket requires a bool, but get {type(ket)}")
        state = np.array(self.sim.get_qs())
        if ket:
            return '\n'.join(ket_string(state))
        return state

    def reset(self):
        """Reset mindquantum simulator to quantum zero state."""
        return self.sim.reset()

    def sampling(
        self,
        circuit: Circuit,
        pr: Union[Dict, ParameterResolver] = None,
        shots: int = 1,
        seed: int = None,
    ):
        """Sample the quantum state."""
        if not circuit.all_measures.map:
            raise ValueError("circuit must have at least one measurement gate.")
        _check_input_type("circuit", Circuit, circuit)
        if self.n_qubits < circuit.n_qubits:
            raise ValueError(f"Circuit has {circuit.n_qubits} qubits, which is more than simulator qubits.")
        _check_int_type("sampling shots", shots)
        _check_value_should_not_less("sampling shots", 1, shots)
        if circuit.parameterized:
            if pr is None:
                raise ValueError("Sampling a parameterized circuit need a ParameterResolver")
            if not isinstance(pr, (dict, ParameterResolver)):
                raise TypeError(f"pr requires a dict or a ParameterResolver, but get {type(pr)}!")
            pr = ParameterResolver(pr)
        else:
            pr = ParameterResolver()
        if seed is None:
            seed = int(np.random.randint(1, 2 << 20))
        else:
            _check_seed(seed)
        res = MeasureResult()
        res.add_measure(circuit.all_measures.keys())
        if circuit.is_measure_end and not circuit.is_noise_circuit:
            sampler = self.sim.sampling_measure_ending_without_nosise
        else:
            sampler = self.sim.sampling
        samples = np.array(sampler(circuit.get_cpp_obj(), pr, shots, res.keys_map, seed)).reshape((shots, -1))
        res.collect_data(samples)
        return res

    def set_qs(self, quantum_state: np.ndarray):
        """Set quantum state of simulator."""
        if isinstance(quantum_state, list):
            quantum_state = np.array(quantum_state)
        if not isinstance(quantum_state, np.ndarray):
            raise TypeError(f"quantum state must be a ndarray, but get {type(quantum_state)}")
        n_qubits = np.log2(quantum_state.shape[0])
        if n_qubits % 1 != 0:
            raise ValueError(f"vec size {quantum_state.shape[0]} is not power of 2")
        n_qubits = int(n_qubits)
        if self.n_qubits != n_qubits:
            raise ValueError(f"{n_qubits} qubits vec does not match with simulation qubits ({self.n_qubits})")
        if self.name == "mqmatrix":
            if len(quantum_state.shape) == 1:
                norm_factor = np.sqrt(np.sum(np.abs(quantum_state) ** 2))
                if norm_factor == 0.0:
                    raise ValueError("Wrong quantum state.")
                self.sim.set_qs(quantum_state / norm_factor)
            elif len(quantum_state.shape) == 2:
                if not np.allclose(quantum_state, quantum_state.T.conj(), atol=1e-6):
                    raise ValueError("density matrix must be hermitian.")
                if (quantum_state.diagonal() < 0).any():
                    raise ValueError("the diagonal terms in density matrix cannot be negative.")
                norm_factor = np.real(np.trace(quantum_state))
                if norm_factor == 0.0:
                    raise ValueError("Wrong quantum state.")
                self.sim.set_dm(quantum_state / norm_factor)
            else:
                raise ValueError(
                    f"vec requires a 1-dimensional array, density matrix requires \
                        a 2-dimensional array, but get {quantum_state.shape}"
                )
        else:
            if len(quantum_state.shape) != 1:
                raise ValueError(f"vec requires a 1-dimensional array, but get {quantum_state.shape}")
            norm_factor = np.sqrt(np.sum(np.abs(quantum_state) ** 2))
            if norm_factor == 0.0:
                raise ValueError("Wrong quantum state.")
            self.sim.set_qs(quantum_state / norm_factor)

    def get_partial_trace(self, obj_qubits) -> np.ndarray:
        """Get partial trace of density matrix."""
        if isinstance(obj_qubits, int):
            obj_qubits = [obj_qubits]
        _check_input_type("obj_qubits", (int, Iterable), obj_qubits)
        if len(set(obj_qubits)) != len(obj_qubits):
            raise ValueError("obj_qubits cannot be repeated.")
        if self.n_qubits <= max(obj_qubits):
            raise ValueError(f"object qubits {obj_qubits} is higher than simulator qubits.")
        if not self.name.startswith('mqmatrix'):
            raise ValueError(f"{self.name} simulator not support partial trace method.")
        return np.array(self.sim.get_partial_trace(obj_qubits))

    def entropy(self) -> float:
        """Get the von-Neumann entropy of quantum state."""
        if self.name.startswith('mqvector'):
            return 0
        if not self.name.startswith('mqmatrix'):
            raise ValueError(f"{self.name} simulator not support entropy method.")
        eigens = np.real(eigvals(self.get_qs()))
        res = 0
        for i in eigens:
            if i > 0:
                res += -i * np.log(i)
        return res

    def purity(self) -> float:
        """Calculate the purity of quantum state."""
        if self.name.startswith('mqvector'):
            return 1.0
        return self.sim.purity()

    def get_pure_state_vector(self) -> np.ndarray:
        """Get the state vector from a pure density matrix."""
        if self.name.startswith('mqvector'):
            return self.get_qs()
        if 1 - self.purity() > 1e-6:
            raise ValueError("Cannot transform mixed density matrix to vector.")
        return np.array(self.sim.pure_state_vector())
