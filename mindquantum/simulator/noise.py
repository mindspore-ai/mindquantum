# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Noise simulator."""
from typing import Dict, Union

import numpy as np

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.circuit.channel_adder import ChannelAdderBase
from mindquantum.core.gates import BasicGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.device.chip import NaiveChip
from mindquantum.simulator.backend_base import BackendBase


# pylint: disable=abstract-method,super-init-not-called,too-many-arguments
class NoiseBackend(BackendBase):
    """Add noise based on channel adder."""

    def __init__(self, base_sim: str, n_qubits: int, adder: ChannelAdderBase, seed: int = None, dtype=mq.complex128):
        """Initialize a noise backend."""
        # pylint: disable=import-outside-toplevel
        from mindquantum.simulator import Simulator

        self.base_sim = Simulator(base_sim, n_qubits, seed=seed)
        self.adder: ChannelAdderBase = adder
        self.name = "NoiseBackend"

    def apply_circuit(self, circuit: Circuit, pr: Union[Dict, ParameterResolver] = None):
        """Apply a quantum circuit."""
        return self.base_sim.apply_circuit(self.adder(circuit), pr)

    def apply_gate(self, gate: BasicGate, pr: Union[Dict, ParameterResolver] = None, diff: bool = False):
        """Apply a quantum gate."""
        if diff:
            raise ValueError("For noise simulator, you cannot set diff to True.")
        # pylint: disable=too-many-function-args
        return self.base_sim.apply_circuit(self.adder(Circuit[gate]), pr, diff)

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """Apply a hamiltonian."""
        return self.base_sim.apply_hamiltonian(hamiltonian)

    def get_expectation(self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None) -> np.ndarray:
        """Get expectation of a hamiltonian."""
        if circ_right is not None:
            circ_right = self.adder(circ_right)
        if circ_left is not None:
            circ_left = self.adder(circ_left)
        return self.base_sim.get_expectation(hamiltonian, circ_right, circ_left, simulator_left, pr)

    def get_qs(self, ket=False):
        """Get quantum state."""
        return self.base_sim.get_qs(ket)

    def reset(self):
        """Reset mindquantum simulator to quantum zero state."""
        self.base_sim.reset()

    def sampling(self, circuit: Circuit, pr: Union[Dict, ParameterResolver] = None, shots: int = 1, seed: int = None):
        """Sample the quantum state."""
        return self.base_sim.sampling(self.adder(circuit), pr, shots, seed)

    def transform_circ(self, circuit: Circuit) -> Circuit:
        """Transform a noiseless circuit to a noise circuit based on this noise backend."""
        return self.adder(circuit)


class ChipBaseBackend(NoiseBackend):
    """
    Add nose based on given chip.

    Topology of device and device supported gate set should be considered.
    """

    def __init__(self, chip: NaiveChip):
        """Initialize base chip."""
        self.chip = chip
