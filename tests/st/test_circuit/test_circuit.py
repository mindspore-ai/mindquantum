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
"""Test circuit."""

from mindquantum.ops import QubitOperator
import numpy as np
from mindquantum import Circuit
import mindquantum.gate as G
from mindquantum.circuit import pauli_word_to_circuits
from mindquantum.circuit import decompose_single_term_time_evolution
from mindquantum.circuit import UN, SwapParts, generate_uccsd
from mindquantum.circuit import TimeEvolution


def test_time_evolution():
    h = QubitOperator('Z0 Z1', 'p')
    circ = TimeEvolution(h).circuit
    circ_exp = Circuit([G.X.on(1, 0), G.RZ({'p': 2}).on(1), G.X.on(1, 0)])
    assert circ == circ_exp


def test_circuit():
    circuit1 = Circuit()
    circuit1 += G.RX('a').on(0)
    circuit1 *= 2
    circuit2 = Circuit([G.X.on(0, 1)])
    circuit3 = circuit1 + circuit2
    assert len(circuit3) == 3
    circuit3.summary(False)
    assert circuit3.n_qubits == 2
    circuit3.insert(0, G.H.on(0))
    assert circuit3[0] == G.H(0)
    circuit3.no_grad()
    assert len(circuit3[1].coeff.requires_grad_parameters) == 0
    circuit3.requires_grad()
    assert len(circuit3[1].coeff.requires_grad_parameters) == 1
    assert len(circuit3.parameter_resolver()) == 1
    assert circuit3.mindspore_data() == {
        'gate_names': ['npg', 'RX', 'RX', 'npg'],
        'gate_matrix': [[[['0.7071067811865475', '0.7071067811865475'],
                          ['0.7071067811865475', '-0.7071067811865475']],
                         [['0.0', '0.0'], ['0.0', '0.0']]],
                        [[['0.0', '0.0'], ['0.0', '0.0']],
                         [['0.0', '0.0'], ['0.0', '0.0']]],
                        [[['0.0', '0.0'], ['0.0', '0.0']],
                         [['0.0', '0.0'], ['0.0', '0.0']]],
                        [[['0.0', '1.0'], ['1.0', '0.0']],
                         [['0.0', '0.0'], ['0.0', '0.0']]]],
        'gate_obj_qubits': [[0], [0], [0], [0]],
        'gate_ctrl_qubits': [[], [], [], [1]],
        'gate_params_names': [[], ['a'], ['a'], []],
        'gate_coeff': [[], [1.0], [1.0], []],
        'gate_requires_grad': [[], [True], [True], []]
    }


def test_circuit_apply():
    circuit = Circuit()
    circuit += G.RX('a').on(0, 1)
    circuit += G.H.on(0)
    circuit = circuit.apply_value({'a': 0.2})
    circuit_exp = Circuit([G.RX(0.2).on(0, 1), G.H.on(0)])
    assert circuit == circuit_exp


def test_pauli_word_to_circuits():
    circ = pauli_word_to_circuits(QubitOperator('Z0 Y1'))
    assert circ == Circuit([G.Z.on(0), G.Y.on(1)])


def test_un():
    circ = UN(G.X, [3, 4, 5], [0, 1, 2])
    assert circ[-1] == G.X.on(5, 2)


def test_swappart():
    circ = SwapParts([1, 2, 3], [4, 5, 6], 0)
    assert circ[-1] == G.SWAP([3, 6], 0)


def test_decompose_single_term_time_evolution():
    circ = decompose_single_term_time_evolution(QubitOperator('Z0 Z1'),
                                                {'a': 1})
    assert circ == Circuit([G.X.on(1, 0), G.RZ({'a': 2}).on(1), G.X.on(1, 0)])


def test_generate_uccsd():
    circ, init_amp, para_name, ham, n_q, n_e = generate_uccsd(
        './tests/st/LiH.hdf5')
    assert len(circ) == 4416
    assert circ[2000] == G.X.on(9, 8)
    assert np.allclose(init_amp[-5], 0.001687182323430231)
    assert len(para_name) == 20
    assert len(ham.terms) == 631
    assert n_q == 12
    assert n_e == 4
