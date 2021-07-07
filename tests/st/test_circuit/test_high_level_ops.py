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
"""Test high level ops."""

import numpy as np
from mindquantum import Circuit, X, H, RX
from mindquantum.circuit import controlled as C
from mindquantum.circuit import dagger as D
from mindquantum.circuit import apply as A
from mindquantum.circuit import AP
from mindquantum.circuit import CPN
from mindquantum.circuit import StateEvolution
from mindquantum.circuit import qft


def test_apply():
    u = unit1([0])
    u2 = A(unit1, [1, 2])
    u2 = u2([0])
    u3 = A(u, [1, 2])
    u_exp = Circuit([X.on(1), H.on(2), RX('a_0').on(1)])
    assert u2 == u3 == u_exp


def test_add_prefix():
    u = unit1([0])
    u2 = AP(u, 'x')
    u3 = AP(unit1, 'x')
    u3 = u3([0])
    u_exp = Circuit([X.on(0), H.on(1), RX('x_a_0').on(0)])
    assert u2 == u3 == u_exp


def test_change_param_name():
    u = unit1([0])
    u2 = CPN(u, {'a_0': 'x'})
    u3 = CPN(unit1, {'a_0': 'x'})
    u3 = u3([0])
    u_exp = Circuit([X.on(0), H.on(1), RX('x').on(0)])
    assert u2 == u3 == u_exp


def unit1(rotate_qubits):
    circuit = Circuit()
    circuit += X.on(0)
    circuit += H.on(1)
    for q in rotate_qubits:
        circuit += RX(f'a_{q}').on(q)
    return circuit


def test_controlled_and_dagger():
    qubits = [0, 1, 2, 3]
    c1 = C(unit1)(4, qubits)
    c2 = C(unit1(qubits))(4)
    assert c1 == c2

    c3 = C(C(unit1))(4, 5, qubits)
    c4 = C(C(unit1)(4, qubits))(5)
    c5 = C(C(unit1(qubits)))(4, 5)
    assert c3 == c4 == c5

    c6 = D(unit1)(qubits)
    c7 = D(unit1(qubits))
    assert c6 == c7

    c8 = D(C(unit1))(4, qubits)
    c9 = C(D(unit1))(4, qubits)
    assert c8 == c9


def test_state_evol():
    qubits = [0, 1, 2, 3]
    circuit = X.on(4) + C(D(unit1(qubits)))(4)
    circuit_exp = Circuit()
    circuit_exp += X.on(4)
    circuit_exp += RX({'a_3': -1}).on(3, 4)
    circuit_exp += RX({'a_2': -1}).on(2, 4)
    circuit_exp += RX({'a_1': -1}).on(1, 4)
    circuit_exp += RX({'a_0': -1}).on(0, 4)
    circuit_exp += H.on(1, 4)
    circuit_exp += X.on(0, 4)
    assert circuit_exp == circuit
    pr = {'a_0': 1, 'a_1': 2, 'a_2': 3, 'a_3': 4}
    fs1 = StateEvolution(circuit).final_state(pr)
    fs2 = StateEvolution(circuit.apply_value(pr)).final_state()
    assert np.allclose(fs1, fs2)
    np.random.seed(42)
    sampling = StateEvolution(circuit.apply_value(pr)).sampling(shots=100)
    sampling_exp = {
        '00000': 0,
        '00001': 0,
        '00010': 0,
        '00011': 0,
        '00100': 0,
        '00101': 0,
        '00110': 0,
        '00111': 0,
        '01000': 0,
        '01001': 0,
        '01010': 0,
        '01011': 0,
        '01100': 0,
        '01101': 0,
        '01110': 0,
        '01111': 0,
        '10000': 0,
        '10001': 0,
        '10010': 0,
        '10011': 0,
        '10100': 0,
        '10101': 2,
        '10110': 0,
        '10111': 2,
        '11000': 0,
        '11001': 0,
        '11010': 0,
        '11011': 0,
        '11100': 7,
        '11101': 44,
        '11110': 4,
        '11111': 41
    }
    assert sampling == sampling_exp


def test_qft():
    c = qft(range(4))
    s = StateEvolution(c).final_state()
    s_exp = np.ones(2**4) * 0.25
    assert np.allclose(s, s_exp)
