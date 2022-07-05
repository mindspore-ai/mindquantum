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
"""Test fermion operator."""

from mindquantum.core.operators import FermionOperator


def test_fermion_ops_num_coeff():
    """
    Description: Test fermion ops num coeff
    Expectation:
    """
    # check the creation operator
    a_p_dagger = FermionOperator('1^')
    assert str(a_p_dagger) == '1 [1^] '

    # check the annihilation operator
    a_q = FermionOperator('0')
    assert str(a_q) == '1 [0] '

    # check zero operator
    zero = FermionOperator()
    assert str(zero) == '0'

    # check identity operator
    identity = FermionOperator('')
    assert str(identity) == '1 [] '


def test_power():
    """
    Description: Test fermion operator power
    Expectation:
    """
    # check power and multiply
    w = (1 + 2j) * FermionOperator(' 4^ 3 9 3^ ') + 4 * FermionOperator(' 2 ')
    w_2 = w * w
    w_3 = w**2
    assert w_2 == w_3


def test_normal_order():
    """
    Description: Test normal order
    Expectation:
    """
    origin = FermionOperator('0 1^')

    normal_order = FermionOperator('1^ 0', -1)

    assert origin.normal_ordered() == normal_order


def test_multiplier():
    """
    Description: Test multiplier
    Expectation:
    """
    origin = FermionOperator('0 1^')
    after_mul = FermionOperator('0 1^', 2)
    assert after_mul == 2 * origin

    # Test in-place multiplier
    origin *= 2
    assert after_mul == origin

    # Test right divide
    new = origin / 2.0
    assert str(new) == '1 [0 1^] '

    # Test in-place divide
    origin /= 2
    assert str(origin) == '1 [0 1^] '


def test_add_sub():
    """
    Description: Test add and sub
    Expectation:
    """
    # Test in place add
    w1 = FermionOperator(' 4^ 3 9 3^ ') + 4 * FermionOperator(' 2 ')
    w2 = 4 * FermionOperator(' 2 ')
    w1 -= w2
    assert str(w1) == '1 [4^ 3 9 3^] '


def test_compress():
    """
    Description: Test compress
    Expectation:
    """
    # test compress
    w1 = FermionOperator('4^ 3') + FermionOperator('2', 1e-9)
    w2 = FermionOperator('4^ 3')
    assert w1.compress() == w2

    a = FermionOperator('0 1^', 'x')
    b = FermionOperator('1^ 0', 'x')
    c = a + b
    d = c.normal_ordered()
    assert d.terms == {}


def test_constant():
    """
    Description: Test constant
    Expectation:
    """
    # test constant
    w1 = FermionOperator('4^ 3 9 3^') + 6.0 * FermionOperator('2 3^') + 2.0 * FermionOperator('')
    assert w1.constant == 2.0


def test_para_operators():
    """
    Description: Test para operators
    Expectation:
    """
    para_op = FermionOperator('0 1^', 'x')
    assert str(para_op) == 'x [0 1^] '

    # test the para with the value
    para_dt = {'x': 2}
    op = para_op.subs(para_dt)
    assert str(op) == '2 [0 1^] '


def test_eq():
    """
    Description: Test equal
    Expectation:
    """
    a = FermionOperator('0 1^', 'x')
    assert a.subs({'x': 1}) == FermionOperator('0 1^')


def test_fermion_operator_iter():
    """
    Description: Test fermion operator iter
    Expectation: success.
    """
    a = FermionOperator('0 1^') + FermionOperator('2^ 3', {"a": -3})
    assert a == sum(list(a))
    b = FermionOperator('0 1^')
    b_exp = [FermionOperator('0'), FermionOperator('1^')]
    for idx, o in enumerate(b.singlet()):
        assert o == b_exp[idx]
    assert b.singlet_coeff() == 1
    assert b.is_singlet


def test_dumps_and_loads():
    """
    Description: Test fermion operator dumps to json and json loads to fermion operator
    Expectation:
    """
    f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
    strings = f.dumps()
    obj = FermionOperator.loads(strings)
    assert obj == f


def test_of_fermion_trans():
    """
    Description: Test transfor fermion operator to openfermion back and force.
    Expectation: success.
    """
    from openfermion import FermionOperator as OFFermionOperator

    ofo_ops = OFFermionOperator("1^ 0", 1)
    mq_ops = FermionOperator('1^ 0', 1)
    assert mq_ops.to_openfermion() == ofo_ops
    print(type(FermionOperator.from_openfermion(ofo_ops)), type(mq_ops))
    assert mq_ops == FermionOperator.from_openfermion(ofo_ops)
