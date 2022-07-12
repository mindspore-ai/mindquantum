#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# pylint: disable=pointless-statement,expression-not-assigned,no-member,useless-suppression

"""An example of using the C++ backend for MindQuantum."""

from projectq.backends import Simulator
from projectq.cengines import MainEngine
from projectq.ops import CNOT, All, H, Measure

from mindquantum.experimental import ops
from mindquantum.experimental.circuit import Circuit
from mindquantum.experimental.simulator import projectq

circuit = Circuit()
q0 = circuit.create_qubit()
q1 = circuit.create_qubit()

sim = projectq.Simulator(2165)

circuit.apply_operator(ops.H(), [q0])
circuit.apply_operator(ops.X(), [q0, q1])


sim.run_circuit(circuit)

# ==============================================================================

eng = MainEngine(backend=Simulator(2165), engine_list=[])
qubits = eng.allocate_qureg(2)

H | qubits[0]
CNOT | (qubits[0], qubits[1])
eng.flush()

# ==============================================================================


def print_simulator(simulator):
    """Print the state vector of a ProjectQ-like simulator."""
    qubit_map, state = simulator.cheat()

    print('Simulator:')
    print(qubit_map)
    print(state)
    print('----------')


print_simulator(sim)

print_simulator(eng.backend)

# ==============================================================================

All(Measure) | qubits
eng.flush()
