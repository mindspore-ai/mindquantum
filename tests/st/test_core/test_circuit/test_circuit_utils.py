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

# pylint: disable=invalid-name
"""Test Circuit utils."""
import numpy as np
import pytest

import mindquantum.core.gates as G
from mindquantum.algorithm.library import qft
from mindquantum.core.circuit import AP, CPN, Circuit, add_prefix
from mindquantum.core.circuit import apply as A
from mindquantum.core.circuit import as_ansatz, as_encoder
from mindquantum.core.circuit import controlled as C
from mindquantum.core.circuit import dagger as D
from mindquantum.core.circuit import (
    decompose_single_term_time_evolution,
    pauli_word_to_circuits,
)
from mindquantum.core.operators import QubitOperator


def test_pauli_word_to_circuits():
    """
    Description: test pauli word to circuits
    Expectation:
    """
    circ = pauli_word_to_circuits(QubitOperator('Z0 Y1'))
    assert circ == Circuit([G.Z.on(0), G.Y.on(1)])


def test_decompose_single_term_time_evolution():
    """
    Description: Test decompose_single_term_time_evolution
    Expectation:
    """
    circ = decompose_single_term_time_evolution(QubitOperator('Z0 Z1'), {'a': 1})
    circ = circ.remove_barrier()
    assert circ == Circuit([G.X.on(1, 0), G.RZ({'a': 2}).on(1), G.X.on(1, 0)])


def test_apply():
    """
    Description:
    Expectation:
    """
    u = unit1([0])
    u2 = A(unit1, [1, 2])
    u2 = u2([0])
    u3 = A(u, [1, 2])
    u_exp = Circuit([G.H.on(1), G.H.on(2), G.RX('a_0').on(1)])
    assert u2 == u3 == u_exp


def test_add_prefix():
    """
    Description:
    Expectation:
    """
    u = unit1([0])
    u2 = AP(u, 'x')
    u3 = AP(unit1, 'x')
    u3 = u3([0])
    u_exp = Circuit([G.H.on(0), G.H.on(1), G.RX('x_a_0').on(0)])
    assert u2 == u3 == u_exp


def test_change_param_name():
    """
    Description:
    Expectation:
    """
    u = unit1([0])
    u2 = CPN(u, {'a_0': 'x'})
    u3 = CPN(unit1, {'a_0': 'x'})
    u3 = u3([0])
    u_exp = Circuit([G.H.on(0), G.H.on(1), G.RX('x').on(0)])
    assert u2 == u3 == u_exp


def unit1(rotate_qubits):
    """
    Description:
    Expectation:
    """
    circuit = Circuit()
    circuit += G.H.on(0)
    circuit += G.H.on(1)
    for qubits in rotate_qubits:
        circuit += G.RX(f'a_{qubits}').on(qubits)
    return circuit


def test_controlled_and_dagger():
    """
    Description:
    Expectation:
    """
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
    """
    Description:
    Expectation:
    """
    qubits = [0, 1, 2, 3]
    circuit = G.H.on(4) + C(D(unit1(qubits)))(4)
    circuit_exp = Circuit()
    circuit_exp += G.H.on(4)
    circuit_exp += G.RX({'a_3': -1}).on(3, 4)
    circuit_exp += G.RX({'a_2': -1}).on(2, 4)
    circuit_exp += G.RX({'a_1': -1}).on(1, 4)
    circuit_exp += G.RX({'a_0': -1}).on(0, 4)
    circuit_exp += G.H.on(1, 4)
    circuit_exp += G.H.on(0, 4)
    assert circuit_exp == circuit
    pr = {'a_0': 1, 'a_1': 2, 'a_2': 3, 'a_3': 4}
    fs1 = circuit.get_qs(pr=pr)
    fs2 = circuit.apply_value(pr).get_qs()
    assert np.allclose(fs1, fs2)


def test_qft():
    """
    Description:
    Expectation:
    """
    c = qft(range(4))
    state = c.get_qs()
    state_exp = np.ones(2**4) * 0.25
    assert np.allclose(state, state_exp)


def test_as_encoder_as_ansatz():
    """
    Description: Test set encoder or ansatz of circuit.
    Expectation: succeed.
    """

    @as_encoder
    def circ1():
        out = Circuit()
        out += G.RX('a').on(0)
        out += G.RX('b').on(0)
        return out

    circ11 = circ1() + as_encoder(Circuit([G.RX('c').on(0)]))
    assert circ11.encoder_params_name == ['a', 'b', 'c']

    @as_ansatz
    def circ2():
        out = Circuit()
        out += G.RX('a').on(0)
        out += G.RX('b').on(0)
        return out

    circ22 = circ2() + as_ansatz(Circuit([G.RX('c').on(0)]))
    assert circ22.ansatz_params_name == ['a', 'b', 'c']

    with pytest.raises(Exception, match="can not be both encoder parameters and ansatz parameters."):
        circ11 + circ22  # pylint: disable=pointless-statement
    res = circ11.as_ansatz() + add_prefix(circ22.as_encoder(), 'e')
    assert res.encoder_params_name == ['e_a', 'e_b', 'e_c']
    assert res.ansatz_params_name == ['a', 'b', 'c']
