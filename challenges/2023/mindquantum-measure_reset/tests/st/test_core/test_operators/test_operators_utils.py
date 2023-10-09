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
import pytest

from mindquantum.core.operators import (
    FermionOperator,
    QubitExcitationOperator,
    QubitOperator,
    TermValue,
    commutator,
    count_qubits,
    down_index,
    hermitian_conjugated,
    normal_ordered,
    number_operator,
    up_index,
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_count_qubits():
    """
    Feature: count_qubits.
    Description: Test count_qubits.
    Expectation: success.
    """
    qubit_op = QubitOperator("X1 Y2")
    assert count_qubits(qubit_op) == 3

    fer_op = FermionOperator("1^")
    assert count_qubits(fer_op) == 2

    qubit_exc_op = QubitExcitationOperator("4^ 1")
    assert count_qubits(qubit_exc_op) == 5


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_normal_ordered():
    """
    Feature: normal_ordered
    Description: Test normal_ordered.
    Expectation: success.
    """
    fermion_operator = FermionOperator("3 4^")
    assert str(normal_ordered(fermion_operator)) == '-1 [4^ 3]'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_commutator():
    """
    Feature: commutator.
    Description: Test commutator.
    Expectation: success.
    """
    qub_op1 = QubitOperator("X1 Y2")
    qub_op2 = QubitOperator("X1 Z2")
    qub_op3 = 2j * QubitOperator("X2")
    assert commutator(qub_op1, qub_op2) == qub_op3

    assert commutator(qub_op1, qub_op1) == QubitOperator()

    qubit_exc_op1 = QubitExcitationOperator(((4, TermValue.adg), (1, TermValue.a)), 2.0j)
    qubit_exc_op2 = QubitExcitationOperator(((3, TermValue.adg), (2, TermValue.a)), 2.0j)
    qubit_exc_op3 = QubitExcitationOperator("3^ 2 4^ 1", 4.0) + QubitExcitationOperator("4^ 1 3^ 2", -4.0)
    assert commutator(qubit_exc_op1, qubit_exc_op2).compress() == qubit_exc_op3

    assert commutator(qubit_exc_op1, qubit_exc_op1) == QubitExcitationOperator()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_number_operator():
    """
    Feature: number operator.
    Description: Test number_operator.
    Expectation: success.
    """
    nmode = 3
    # other parameters by default
    check_str = '1 [0^ 0] +\n1 [1^ 1] +\n1 [2^ 2]'
    assert str(number_operator(nmode)) == check_str

    check_str2 = '1 [3^ 3]'
    assert str(number_operator(None, nmode)) == check_str2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_up_index():
    """
    Feature: up index.
    Description: Test labelling the spin-orbital index with spin beta.
    Expectation: success.
    """
    alpha = 2
    assert up_index(alpha) == 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_down_index():
    """
    Feature: down index.
    Description: Test labelling the spin-orbital index with spin beta.
    Expectation: success.
    """
    beta = 1
    assert down_index(beta) == 3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_hermitian_conjugated():
    """
    Feature: hermitian conjugated.
    Description: Test hermitian_conjugated for the QubitOperator and Fermion Operator.
    Expectation: success.
    """
    qub_op1 = -1j * QubitOperator("X1 Y2") + QubitOperator("X1")
    qub_op2 = 1j * QubitOperator("X1 Y2") + QubitOperator("X1")

    assert hermitian_conjugated(qub_op1) == qub_op2

    fer_op1 = FermionOperator("1^ 2")
    fer_op2 = FermionOperator("2^ 1")
    assert hermitian_conjugated(fer_op1) == fer_op2

    qubit_exc_op1 = QubitExcitationOperator(((4, TermValue.adg), (1, TermValue.a)), 2.0j).normal_ordered()
    qubit_exc_op2 = QubitExcitationOperator(((4, TermValue.a), (1, TermValue.adg)), -2.0j).normal_ordered()
    assert hermitian_conjugated(qubit_exc_op1) == qubit_exc_op2
