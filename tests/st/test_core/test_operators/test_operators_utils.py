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
"""Test operator_utils."""

from mindquantum.core.operators import (
    FermionOperator,
    QubitExcitationOperator,
    QubitOperator,
)
from mindquantum.core.operators.utils import (
    commutator,
    count_qubits,
    down_index,
    hermitian_conjugated,
    normal_ordered,
    number_operator,
    up_index,
)


def test_count_qubits():
    """Test count_qubits"""
    qubit_op = QubitOperator("X1 Y2")
    assert count_qubits(qubit_op) == 3

    fer_op = FermionOperator("1^")
    assert count_qubits(fer_op) == 2

    qubit_exc_op = QubitExcitationOperator("4^ 1")
    assert count_qubits(qubit_exc_op) == 5


def test_normal_ordered():
    """Test normal_ordered function"""
    op = FermionOperator("3 4^")
    assert str(normal_ordered(op)) == '-1 [4^ 3] '


def test_commutator():
    """Test commutator"""
    qub_op1 = QubitOperator("X1 Y2")
    qub_op2 = QubitOperator("X1 Z2")
    qub_op3 = 2j * QubitOperator("X2")
    assert commutator(qub_op1, qub_op2) == qub_op3

    assert commutator(qub_op1, qub_op1) == QubitOperator()

    qubit_exc_op1 = QubitExcitationOperator(((4, 1), (1, 0)), 2.0j)
    qubit_exc_op2 = QubitExcitationOperator(((3, 1), (2, 0)), 2.0j)
    qubit_exc_op3 = QubitExcitationOperator("3^ 2 4^ 1", 4.0) + QubitExcitationOperator("4^ 1 3^ 2", -4.0)
    assert commutator(qubit_exc_op1, qubit_exc_op2).compress() == qubit_exc_op3

    assert commutator(qubit_exc_op1, qubit_exc_op1) == QubitExcitationOperator()


def test_number_operator():
    """Test number operator"""
    nmode = 3
    # other parameters by default
    check_str = '1 [0^ 0] +\n1 [1^ 1] +\n1 [2^ 2] '
    assert str(number_operator(nmode)) == check_str

    check_str2 = '1 [3^ 3] '
    assert str(number_operator(None, nmode)) == check_str2


def test_up_index():
    """This is for labelling the spin-orbital index with spin alpha"""
    alpha = 2
    assert up_index(alpha) == 4


def test_down_index():
    """This is for labelling the spin-orbital index with spin beta"""
    beta = 1
    assert down_index(beta) == 3


def test_hermitian_conjugated():
    """Test hermitian_conjugated for the QubitOperator and Fermion Operator"""
    qub_op1 = -1j * QubitOperator("X1 Y2") + QubitOperator("X1")
    qub_op2 = 1j * QubitOperator("X1 Y2") + QubitOperator("X1")

    assert hermitian_conjugated(qub_op1) == qub_op2

    fer_op1 = FermionOperator("1^ 2")
    fer_op2 = FermionOperator("2^ 1")
    assert hermitian_conjugated(fer_op1) == fer_op2

    qubit_exc_op1 = QubitExcitationOperator(((4, 1), (1, 0)), 2.0j).normal_ordered()
    qubit_exc_op2 = QubitExcitationOperator(((4, 0), (1, 1)), -2.0j).normal_ordered()
    assert hermitian_conjugated(qubit_exc_op1) == qubit_exc_op2
