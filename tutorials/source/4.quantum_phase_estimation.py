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

# pylint: disable=invalid-name,duplicate-code

"""Example of performing a quantum phase estimation algorithm."""

import numpy as np

from mindquantum import BARRIER, UN, Circuit, H, Measure, PhaseShift, Simulator, X, qft

n_qubits = 3
c = Circuit()
c += UN(H, n_qubits)
c += X.on(n_qubits)
print(c)

for i in range(n_qubits):
    c += PhaseShift({'phi': 2**i}).on(n_qubits, n_qubits - i - 1)
print(c)

c += BARRIER
c += qft(range(n_qubits)).hermitian()
print(c)


sim = Simulator('projectq', c.n_qubits)
phi = 0.125
sim.apply_circuit(c, {'phi': 2 * np.pi * phi})
qs = sim.get_qs()
print(sim.get_qs(ket=True))
res = sim.sampling(UN(Measure(), c.n_qubits), shots=100)
print(res)

index = np.argmax(np.abs(qs))
print(index)

bit_string = bin(index)[2:].zfill(c.n_qubits)[1:]
print(bit_string)

theta_exp = int(bit_string[::-1], 2) / 2**n_qubits
print(theta_exp)
