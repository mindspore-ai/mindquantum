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
"""Backend base."""

from typing import Dict, List, Union

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BasicGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.utils.type_value_check import _check_seed


class BackendBase:
    """Backend interface."""

    def __init__(self, name: str, n_qubits: int, seed=None, dtype=None):
        """Initialize backend obj."""
        if seed is None:
            seed = np.random.randint(1, 2**23)
        _check_seed(seed)
        self.name = name
        self.n_qubits = n_qubits
        self.seed = seed
        self.arithmetic_type = dtype

    def apply_circuit(
        self,
        circuit: Circuit,
        pr: Union[Dict, ParameterResolver] = None,
    ):
        """Apply a quantum circuit."""
        raise NotImplementedError(f"apply_circuit not implemented for {self.device_name()}")

    def apply_gate(
        self,
        gate: BasicGate,
        pr: Union[Dict, ParameterResolver] = None,
        diff: bool = False,
    ):
        """Apply a quantum gate."""
        raise NotImplementedError(f"apply_gate not implemented for {self.device_name()}")

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """Apply a hamiltonian."""
        raise NotImplementedError(f"apply_hamiltonian not implemented for {self.device_name()}")

    def astype(self, dtype, seed):
        """Convert simulator to other data type."""
        raise NotImplementedError(f"astype not implement for {self.device_name()}")

    @property
    def dtype(self):
        """Get data type of simulator."""
        raise NotImplementedError(f"get dtype not implement for {self.device_name()}")

    def copy(self) -> "BackendBase":
        """Copy this backend."""
        raise NotImplementedError(f"copy not implemented for {self.device_name()}")

    def device_name(self) -> str:
        """Return the device name of this backend."""
        return self.name

    def get_circuit_matrix(self, circuit: Circuit, pr: ParameterResolver) -> np.ndarray:
        """Get the matrix of given circuit."""
        raise NotImplementedError(f"get_circuit_matrix not implemented for {self.device_name()}")

    # pylint: disable=too-many-arguments
    def get_expectation(self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None) -> np.ndarray:
        """Get expectation of given hamiltonian."""
        raise NotImplementedError(f"get_expectation not implemented for {self.device_name()}")

    def get_expectation_with_grad(  # pylint: disable=too-many-arguments
        self,
        hams: List[Hamiltonian],
        circ_right: Circuit,
        circ_left: Circuit = None,
        simulator_left: "BackendBase" = None,
        parallel_worker: int = None,
        pr_shift: bool = False,
    ):
        """Get expectation and the gradient w.r.t parameters."""
        raise NotImplementedError(f"get_expectation_with_grad not implemented for {self.device_name()}")

    def get_qs(self, ket=False) -> Union[str, np.ndarray]:
        """Get quantum state."""
        raise NotImplementedError(f"get_qs not implemented for {self.device_name()}")

    def reset(self):
        """Reset backend to quantum zero state."""
        raise NotImplementedError(f"reset not implemented for {self.device_name()}")

    def sampling(
        self,
        circuit: Circuit,
        pr: Union[Dict, ParameterResolver] = None,
        shots: int = 1,
        seed: int = None,
    ):
        """Sample a quantum state based on this backend."""
        raise NotImplementedError(f"sampling not implemented for {self.device_name()}")

    def set_qs(self, quantum_state: np.ndarray):
        """Set quantum state of this backend."""
        raise NotImplementedError(f"set_qs not implemented for {self.device_name()}")

    def set_threads_number(self, number):
        """Set maximum number of threads."""
        raise NotImplementedError(f"set_threads_number not implemented for {self.device_name()}")

    def get_partial_trace(self, obj_qubits):
        """Get partial trace of density matrix."""
        raise NotImplementedError(f"get_partial_trace not implemented for {self.device_name()}")

    def entropy(self):
        """Get the von-Neumann entropy of quantum state."""
        raise NotImplementedError(f"entropy not implemented for {self.device_name()}")

    def purity(self):
        """Calculate the purity of quantum state."""
        raise NotImplementedError(f"purity not implemented for {self.device_name()}")

    def get_pure_state_vector(self) -> np.ndarray:
        """Get the state vector from a pure density matrix."""
        raise NotImplementedError(f"get_pure_state_vector not implemented for {self.device_name()}")
