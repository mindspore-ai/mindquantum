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

# pylint: disable=pointless-statement,expression-not-assigned

"""Test circuitengine."""

import mindquantum.core.gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.engine import circuit_generator
from mindquantum.engine.circuitengine import CircuitEngine


def test_allocate_qureg():
    """Test allocate qureg."""
    eng = CircuitEngine()
    qubits = eng.allocate_qureg(2)
    G.H | (eng.qubits[0],)
    G.X | (eng.qubits[0], eng.qubits[1])
    assert qubits[0].qubit_id == 0
    assert qubits[1].qubit_id == 1


def test_circuit_generator():
    """Test circuit generator."""

    @circuit_generator(3)
    def encoder(qubits):
        G.H | (qubits[0],)
        G.X | (qubits[0], qubits[1])
        G.RY('p1') | (qubits[2],)

    assert isinstance(encoder, Circuit)
