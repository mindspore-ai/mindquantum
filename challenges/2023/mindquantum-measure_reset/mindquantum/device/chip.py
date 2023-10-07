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
"""Real device chip."""
from ..core.circuit import Circuit
from . import QubitNode, QubitsTopology
from .vigo_property import vigo_noise_model


# pylint: disable=too-few-public-methods
class NaiveChip:
    """A naive quantum chip."""

    def __init__(self, topology: QubitsTopology):
        """Initialize a naive quantum chip."""
        self.topology = topology


class Vigo(NaiveChip):
    """5 qubit chip named Vigo."""

    def __init__(self):
        """Construct vigo chip."""
        topology = QubitsTopology([QubitNode(i) for i in range(5)])
        _ = topology[0] >> topology[1] >> topology[3] >> topology[4]
        _ = topology[1] >> topology[2]
        topology[0].set_poi(0, 0)
        topology[1].set_poi(1, 0)
        topology[2].set_poi(2, 0)
        topology[3].set_poi(1, 1)
        topology[4].set_poi(1, 2)
        super().__init__(topology)
        self.noise_model = vigo_noise_model

    def gene_noise_circuit(self, circ):
        """
        Generate noise circuit.

        Args:
            circ (Circuit): quantum circuit.

        Returns:
            Circuit: circuit with noise.
        """
        noise_circuit = Circuit()
        for i in range(circ.n_qubits):
            noise_circuit += self.noise_model.get(('prepare', i), Circuit())
        noise_circuit.barrier(False)
        for g in circ:
            noise_circuit += g
            noise_circuit += self.noise_model.get((g.name, tuple(g.obj_qubits + g.ctrl_qubits)), Circuit())
        noise_circuit.barrier(False)
        for i in range(circ.n_qubits):
            noise_circuit += self.noise_model.get(('readout', i), Circuit())
        return noise_circuit
