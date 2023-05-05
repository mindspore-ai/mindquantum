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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# pylint: disable=invalid-name
"""Test qubit excitation operator."""
import pickle

import pytest

from mindquantum.core.operators import (
    FermionOperator,
    QubitExcitationOperator,
    QubitOperator,
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qubit_excitation_ops_num_coeff():
    """
    Description: check the creation operator
    Expectation:
    """
    a_p_dagger = QubitExcitationOperator('1^')
    assert str(a_p_dagger) == '1 [Q1^] '

    # check the annihilation operator
    a_q = QubitExcitationOperator('0')
    assert str(a_q) == '1 [Q0] '

    # check zero operator
    zero = QubitExcitationOperator()
    assert str(zero) == '0'

    # check identity operator
    identity = QubitExcitationOperator('')
    assert str(identity) == '1 [] '


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_power():
    """
    Description: check power and multiply
    Expectation:
    """
    w = (1 + 2j) * QubitExcitationOperator(' 4^ 3 9 3^ ') + 4 * QubitExcitationOperator(' 2 ')
    w_2 = w * w
    w_3 = w**2
    assert w_2 == w_3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_normal_order():
    """
    Description: Test normal order
    Expectation:
    """
    origin = QubitExcitationOperator('0 1^')
    # Coefficient will not be affected for qubit-excitation operators
    normal_order = QubitExcitationOperator('1^ 0', 1)

    assert origin.normal_ordered() == normal_order


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_multiplier():
    """
    Description: Test multiplier
    Expectation:
    """
    origin = QubitExcitationOperator('0 1^')
    after_mul = QubitExcitationOperator('0 1^', 2)
    assert after_mul == 2 * origin

    # Test in-place multiplier
    origin *= 2
    assert after_mul == origin

    # Test right divide
    new = origin / 2.0
    assert str(new) == '1 [Q0 Q1^] '

    # Test in-place divide
    origin /= 2
    assert str(origin) == '1 [Q0 Q1^] '


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_add_sub():
    """
    Description: Test add ans sub
    Expectation:
    """
    # Test in place add
    w1 = QubitExcitationOperator(' 4^ 3 9 3^ ') + 4 * QubitExcitationOperator(' 2 ')
    w2 = 4 * QubitExcitationOperator(' 2 ')
    w1 -= w2
    assert str(w1) == '1 [Q4^ Q3 Q9 Q3^] '


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_compress():
    """
    Description: Test compress
    Expectation:
    """
    w1 = QubitExcitationOperator('4^ 3') + QubitExcitationOperator('2', 1e-9)
    w2 = QubitExcitationOperator('4^ 3')
    assert w1.compress() == w2

    a = QubitExcitationOperator('0 1^', 'x')
    b = QubitExcitationOperator('1^ 0', 'x')
    c = a + -b
    d = c.normal_ordered()
    assert d.terms == {}


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_constant():
    """
    Description: Test constant
    Expectation:
    """
    w1 = (
        QubitExcitationOperator('4^ 3 9 3^') + 6.0 * QubitExcitationOperator('2 3^') + 2.0 * QubitExcitationOperator('')
    )
    assert w1.constant == 2.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_para_operators():
    """
    Description: Test para operators
    Expectation:
    """
    para_op = QubitExcitationOperator('0 1^', 'x')
    assert str(para_op) == 'x [Q0 Q1^] '

    # test the para with the value
    para_dt = {'x': 2}
    op = para_op.subs(para_dt)
    assert str(op) == '2 [Q0 Q1^] '


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_eq():
    """
    Description: Test equal
    Expectation:
    """
    a = QubitExcitationOperator('0 1^', 'x')
    assert a.subs({'x': 1}) == QubitExcitationOperator('0 1^')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_convert_to_qubit_operator():
    """
    Description: Check if the qubit excitation operator can correctly convert to
        the qubit operator correctly according to the definition.
    Expectation:
    """
    op = QubitExcitationOperator(((4, 1), (1, 0)), 2.0j)
    qubit_op = (
        QubitOperator("X1 X4", 0.5j)
        + QubitOperator("Y1 X4", -0.5)
        + QubitOperator("X1 Y4", 0.5)
        + QubitOperator("Y1 Y4", 0.5j)
    )
    assert op.to_qubit_operator().compress() == qubit_op


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_fermion_op():
    """
    Description: Test the "Fermion excitation version" of a qubit excitation operator
    Expectation:
    """
    op = QubitExcitationOperator(((4, 1), (1, 0)), 2.0j)
    ferm_op = FermionOperator(((4, 1), (1, 0)), 2.0j)

    assert op.fermion_operator == ferm_op


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_pickle_qubit_excitation():
    """
    Description: Test pickle for qubit excitation operator
    Expectation: success.
    """
    ops = QubitExcitationOperator(((4, 1), (1, 0)), 2.0j)
    assert ops == pickle.loads(pickle.dumps(ops))
