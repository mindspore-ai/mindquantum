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
"""Real device chip."""
from mindquantum.device import QubitsTopology, QubitNode
from mindquantum.device.vigo_property import vigo_noise_model
from mindquantum.core.circuit import Circuit


class NaiveChip:
    def __init__(self, topology: QubitsTopology):
        self.topology = topology

    def gene_channel(self, g, noise_type, with_ctrl, alternative):
        pass


class Vigo(NaiveChip):
    """5 qubit chip named Vigo."""

    def __init__(self):
        topology = QubitsTopology([QubitNode(i) for i in range(5)])
        _ = topology[0] >> topology[1] >> topology[3] >> topology[4]
        _ = topology[1] >> topology[2]
        super().__init__(topology)
        self.noise_model = vigo_noise_model

    def gene_noise_circuit(self, circ):
        """
        generate noise circuit.

        Args:
            circ (Circuit): quantum circuit.

        Returns:
            Circuit: circuit with noise.
        """
        noise_circuit = Circuit()
        for i in range(circ.n_qubits):
            noise_circuit += self.noise_model.get(('prepare', i), Circuit())
        for g in circ:
            noise_circuit += self.noise_model.get((g.name, tuple(g.obj_qubits)), Circuit())
        for i in range(circ.n_qubits):
            noise_circuit += self.noise_model.get(('readout', i), Circuit())
        return noise_circuit
