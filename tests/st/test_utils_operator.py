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
"""Test utils_operator."""

from mindquantum.ops import QubitOperator, FermionOperator
from mindquantum.utils import (count_qubits, normal_ordered, commutator,
                               number_operator, up_index, down_index,
                               hermitian_conjugated)


def test_count_qubits():
    """Test count_qubits"""
    qubit_op = QubitOperator("X1 Y2")
    assert count_qubits(qubit_op) == 3

    fer_op = FermionOperator("1^")
    assert count_qubits(fer_op) == 2


def test_normal_ordered():
    """Test normal_ordered function"""
    op = FermionOperator("3 4^")
    assert str(normal_ordered(op)) == '-1.0 [4^ 3] '


def test_commutator():
    """Test commutator"""
    qub_op1 = QubitOperator("X1 Y2")
    qub_op2 = QubitOperator("X1 Z2")
    qub_op3 = 2j * QubitOperator("X2")
    assert commutator(qub_op1, qub_op2) == qub_op3

    assert commutator(qub_op1, qub_op1) == QubitOperator()


def test_number_operator():
    """Test number operator"""
    nmode = 3
    # other parameters by default
    check_str = '1.0 [0^ 0] +\n1.0 [1^ 1] +\n1.0 [2^ 2] '
    assert str(number_operator(nmode)) == check_str


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
