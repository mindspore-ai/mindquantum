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
import pytest

from mindquantum.config import set_context
from mindquantum.core.operators import (
    FermionOperator,
    QubitExcitationOperator,
    QubitOperator,
)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_qubit_excitation_ops_num_coeff(dtype):
    """
    Description: check the creation operator
    Expectation:
    """
    set_context(dtype=dtype)
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
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_power(dtype):
    """
    Description: check power and multiply
    Expectation:
    """
    set_context(dtype=dtype)
    w = (1 + 2j) * QubitExcitationOperator(' 4^ 3 9 3^ ') + 4 * QubitExcitationOperator(' 2 ')
    w_2 = w * w
    w_3 = w**2
    assert w_2 == w_3


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_normal_order(dtype):
    """
    Description: Test normal order
    Expectation:
    """
    set_context(dtype=dtype)
    origin = QubitExcitationOperator('0 1^')
    # Coefficient will not be affected for qubit-excitation operators
    normal_order = QubitExcitationOperator('1^ 0', 1)

    assert origin.normal_ordered() == normal_order


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_multiplier(dtype):
    """
    Description: Test multiplier
    Expectation:
    """
    set_context(dtype=dtype)
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
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_add_sub(dtype):
    """
    Description: Test add ans sub
    Expectation:
    """
    set_context(dtype=dtype)
    # Test in place add
    w1 = QubitExcitationOperator(' 4^ 3 9 3^ ') + 4 * QubitExcitationOperator(' 2 ')
    w2 = 4 * QubitExcitationOperator(' 2 ')
    w1 -= w2
    assert str(w1) == '1 [Q4^ Q3 Q9 Q3^] '


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_compress(dtype):
    """
    Description: Test compress
    Expectation:
    """
    set_context(dtype=dtype)
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
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_constant(dtype):
    """
    Description: Test constant
    Expectation:
    """
    set_context(dtype=dtype)
    w1 = (
        QubitExcitationOperator('4^ 3 9 3^') + 6.0 * QubitExcitationOperator('2 3^') + 2.0 * QubitExcitationOperator('')
    )
    assert w1.constant == 2.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_para_operators(dtype):
    """
    Description: Test para operators
    Expectation:
    """
    set_context(dtype=dtype)
    para_op = QubitExcitationOperator('0 1^', 'x')
    assert str(para_op) == 'x [Q0 Q1^] '

    # test the para with the value
    para_dt = {'x': 2}
    op = para_op.subs(para_dt)
    assert str(op) == '2 [Q0 Q1^] '


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_eq(dtype):
    """
    Description: Test equal
    Expectation:
    """
    set_context(dtype=dtype)
    a = QubitExcitationOperator('0 1^', 'x')
    assert a.subs({'x': 1}) == QubitExcitationOperator('0 1^')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_convert_to_qubit_operator(dtype):
    """
    Description: Check if the qubit excitation operator can correctly convert to
        the qubit operator correctly according to the definition.
    Expectation:
    """
    set_context(dtype=dtype)
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
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_fermion_op(dtype):
    """
    Description: Test the "Fermion excitation version" of a qubit excitation operator
    Expectation:
    """
    set_context(dtype=dtype)
    op = QubitExcitationOperator(((4, 1), (1, 0)), 2.0j)
    ferm_op = FermionOperator(((4, 1), (1, 0)), 2.0j)

    assert op.fermion_operator == ferm_op
