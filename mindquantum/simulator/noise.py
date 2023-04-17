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

import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.circuit.channel_adder import ChannelAdderBase
from mindquantum.core.gates import BasicGate
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.device.chip import NaiveChip
from mindquantum.simulator.backend_base import BackendBase


class NoiseBackend(BackendBase):
    """Add noise based on channel adder."""

    def __init__(self, base_sim: str, n_qubits: int, adder: ChannelAdderBase, seed: int = 42, dtype=mq.complex128):
        from mindquantum.simulator import Simulator

        self.base_sim = Simulator(base_sim, n_qubits, seed)
        self.adder: ChannelAdderBase = adder

    def apply_circuit(self, circuit: Circuit, pr: Union[Dict, ParameterResolver] = None):
        return self.base_sim.apply_circuit(self.adder(circuit), pr)

    def apply_gate(self, gate: BasicGate, pr: Union[Dict, ParameterResolver] = None, diff: bool = False):
        if diff:
            raise ValueError("For noise simulator, you cannot set diff to True.")
        return self.base_sim.apply_circuit(self.adder(Circuit[gate]), pr, diff)

    def sampling(self, circuit: Circuit, pr: Union[Dict, ParameterResolver] = None, shots: int = 1, seed: int = None):
        return self.base_sim.sampling(self.adder(circuit), pr, shots, seed)


class ChipBaseBackend(NoiseBackend):
    """
    Add nose based on given chip.

    Topology of device and device supported gate set should be considered.
    """

    def __init__(self, chip: NaiveChip):
        self.chip = chip
