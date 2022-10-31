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
from typing import Dict, List, Union

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BarrierGate, BasicGate, Measure, MeasureResult
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_ansatz,
    _check_encoder,
    _check_hamiltonian_qubits_number,
    _check_input_type,
    _check_int_type,
    _check_seed,
    _check_value_should_not_less,
)

# This import is required to register some of the C++ types (e.g. ParameterResolver)
from .. import mqbackend  # noqa: F401  # pylint: disable=unused-import
from ..utils.string_utils import ket_string
from .backend_base import BackendBase
from .utils import GradOpsWrapper, _thread_balance

# isort: split

from mindquantum import _mq_vector  # pylint: disable=wrong-import-order

try:
    from mindquantum import _mq_vector_gpu

    MQ_SIM_GPU_SUPPORTED = True
except ImportError:
    MQ_SIM_GPU_SUPPORTED = False


# pylint: disable=abstract-method
class MQSim(BackendBase):
    """Mindquantum Backend."""

    def __init__(self, name: str, n_qubits: int, seed=42):
        """Initialize a mindquantum backend."""
        super().__init__(name, n_qubits, seed)
        if name == 'mqvector':
            self.sim = _mq_vector.mqvector(n_qubits, seed)
        elif name == 'mqvector_gpu':
            if MQ_SIM_GPU_SUPPORTED:
                self.sim = _mq_vector_gpu.mqvector(n_qubits, seed)
        else:
            raise NotImplementedError(f"{name} backend not implemented.")

    def __str__(self):
        """Return a string representation of the object."""
        state = self.get_qs()
        ret = f"{self.name} simulator with {self.n_qubits} qubit{'s' if self.n_qubits > 1 else ''} (little endian)."
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
        res = self.sim.apply_circuit(circuit.get_cpp_obj(), pr.get_cpp_obj())
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
        if not isinstance(gate, BarrierGate):
            gate_max = max(max(gate.obj_qubits, gate.ctrl_qubits))
            if self.n_qubits < gate_max:
                raise ValueError(f"qubits of gate {gate} is higher than simulator qubits.")
            if gate.parameterized:
                if pr is None:
                    raise ValueError("apply a parameterized gate needs a parameter_resolver")
                pr = _check_and_generate_pr_type(pr, gate.coeff.params_name)
            else:
                pr = ParameterResolver()
            if isinstance(gate, Measure):
                return self.sim.apply_gate(gate.get_cpp_obj(), pr.get_cpp_obj(), diff)
            self.sim.apply_gate(gate.get_cpp_obj(), pr.get_cpp_obj(), diff)
        return None

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """Apply a hamiltonian."""
        _check_input_type('hamiltonian', Hamiltonian, hamiltonian)
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)
        self.sim.apply_hamiltonian(hamiltonian.get_cpp_obj())

    def copy(self) -> "BackendBase":
        """Copy a projectq simulator."""
        sim = MQSim(self.name, self.n_qubits, self.seed)
        sim.sim = self.sim.copy()
        return sim

    def device_name(self) -> str:
        """Return the device name."""
        return f"{self.n_qubits} qubits {self.name} simulator."

    def flush(self):
        """Execute all command."""

    def get_circuit_matrix(self, circuit: Circuit, pr: ParameterResolver) -> np.ndarray:
        """Get the matrix of given circuit."""
        return np.array(self.sim.get_circuit_matrix(circuit.get_cpp_obj(), pr.get_cpp_obj())).T

    def get_expectation(self, hamiltonian: Hamiltonian) -> np.ndarray:
        """Get expectation of a hamiltonian."""
        if not isinstance(hamiltonian, Hamiltonian):
            raise TypeError(f"hamiltonian requires a Hamiltonian, but got {type(hamiltonian)}")
        _check_hamiltonian_qubits_number(hamiltonian, self.n_qubits)
        return self.sim.get_expectation(hamiltonian.get_cpp_obj())

    def get_expectation_with_grad(  # pylint: disable=R0912,R0913,R0914,R0915
        self,
        hams: List[Hamiltonian],
        circ_right: Circuit,
        circ_left: Circuit = None,
        simulator_left: "BackendBase" = None,
        parallel_worker: int = None,
    ):
        """Get expectation with grad."""
        if isinstance(hams, Hamiltonian):
            hams = [hams]
        elif not isinstance(hams, list):
            raise TypeError(f"hams requires a Hamiltonian or a list of Hamiltonian, but get {type(hams)}")
        for h_tmp in hams:
            _check_input_type("hams's element", Hamiltonian, h_tmp)
            _check_hamiltonian_qubits_number(h_tmp, self.n_qubits)
        _check_input_type("circ_right", Circuit, circ_right)
        if circ_right.is_noise_circuit:
            raise ValueError("noise circuit not support yet.")
        non_hermitian = False
        if circ_left is not None:
            _check_input_type("circ_left", Circuit, circ_left)
            if circ_left.is_noise_circuit:
                raise ValueError("noise circuit not support yet.")
            non_hermitian = True
        if simulator_left is not None:
            _check_input_type("simulator_left", MQSim, simulator_left)
            if self.name != simulator_left.name:
                raise ValueError(
                    "simulator_left should have the same backend as this simulator, ",
                    f"which is {self.name}, but get {simulator_left.name}",
                )
            if self.n_qubits != simulator_left.n_qubits:
                raise ValueError(
                    "simulator_left should have the same n_qubits as this simulator, ",
                    f"which is {self.n_qubits}, but get {simulator_left.n_qubits}",
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

        def grad_ops(*inputs):
            if version == "both" and len(inputs) != 2:
                raise ValueError("Need two inputs!")
            if version in ("encoder", "ansatz") and len(inputs) != 1:
                raise ValueError("Need one input!")
            if version == "both":
                _check_encoder(inputs[0], len(encoder_params_name))
                _check_ansatz(inputs[1], len(ansatz_params_name))
                batch_threads, mea_threads = _thread_balance(inputs[0].shape[0], len(hams), parallel_worker)
                inputs0 = inputs[0]
                inputs1 = inputs[1]
            if version == "encoder":
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
            grad_ops, hams, circ_right, circ_left, encoder_params_name, ansatz_params_name, parallel_worker
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
        sim = self
        if circuit.is_measure_end and not circuit.is_noise_circuit:
            sim = self.copy()
            sim.apply_circuit(circuit.remove_measure(), pr)
            circuit = Circuit(circuit.all_measures.keys())
        samples = np.array(
            sim.sim.sampling(circuit.get_cpp_obj(), pr.get_cpp_obj(), shots, res.keys_map, seed)
        ).reshape((shots, -1))
        res.collect_data(samples)
        return res

    def set_qs(self, quantum_state: np.ndarray):
        """Set quantum state of mqvector simulator."""
        if not isinstance(quantum_state, np.ndarray):
            raise TypeError(f"quantum state must be a ndarray, but get {type(quantum_state)}")
        if len(quantum_state.shape) != 1:
            raise ValueError(f"vec requires a 1-dimensional array, but get {quantum_state.shape}")
        n_qubits = np.log2(quantum_state.shape[0])
        if n_qubits % 1 != 0:
            raise ValueError(f"vec size {quantum_state.shape[0]} is not power of 2")
        n_qubits = int(n_qubits)
        if self.n_qubits != n_qubits:
            raise ValueError(f"{n_qubits} qubits vec does not match with simulation qubits ({self.n_qubits})")
        self.sim.set_qs(quantum_state / np.sqrt(np.sum(np.abs(quantum_state) ** 2)))
