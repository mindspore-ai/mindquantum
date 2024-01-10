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
"""Test fermion operator."""
import os
import pickle

import numpy as np
import pytest

from mindquantum.core.operators import FermionOperator

_HAS_OPENFERMION = True
try:
    from openfermion import FermionOperator as OFFermionOperator
except (ImportError, AttributeError):
    _HAS_OPENFERMION = False
_FORCE_TEST = bool(os.environ.get("FORCE_TEST", False))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_fermion_ops_num_coeff():
    """
    Description: Test fermion ops num coeff
    Expectation:
    """
    # check the creation operator
    a_p_dagger = FermionOperator('1^')
    assert str(a_p_dagger) == '1 [1^]'

    # check the annihilation operator
    a_q = FermionOperator('0')
    assert str(a_q) == '1 [0]'

    # check zero operator
    zero = FermionOperator()
    assert str(zero) == '0'

    # check identity operator
    identity = FermionOperator('')
    assert str(identity) == '1 []'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_normal_order():
    """
    Description: Test normal order
    Expectation:
    """
    origin = FermionOperator('12 13^')

    normal_order = FermionOperator('13^ 12', -1)

    assert origin.normal_ordered() == normal_order


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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
    assert str(new) == '-1 [1^ 0]'

    # Test in-place divide
    origin /= 2
    assert str(origin) == '-1 [1^ 0]'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_add_sub():
    """
    Description: Test add and sub
    Expectation:
    """
    # Test in place add
    w1 = FermionOperator(' 4^ 3 9 3^ ') + 4 * FermionOperator(' 2 ')
    w2 = 4 * FermionOperator(' 2 ')
    w1 -= w2
    assert str(w1) == '1 [9 4^ 3 3^]'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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
    assert not d.terms


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_constant():
    """
    Description: Test constant
    Expectation:
    """
    # test constant
    w1 = FermionOperator('4^ 3 9 3^') + 6.0 * FermionOperator('2 3^') + 2.0 * FermionOperator('')
    assert w1.constant == 2.0


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_para_operators():
    """
    Description: Test para operators
    Expectation:
    """
    para_op = FermionOperator('0 1^', 'x')
    assert str(para_op) == '-x [1^ 0]'

    # test the para with the value
    para_dt = {'x': 2}
    op = para_op.subs(para_dt)
    assert str(op) == '-2 [1^ 0]'


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_eq():
    """
    Description: Test equal
    Expectation:
    """
    a = FermionOperator('0 1^', 'x')
    assert a.subs({'x': 1}) == FermionOperator('0 1^')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_fermion_operator_iter():
    """
    Description: Test fermion operator iter
    Expectation: success.
    """
    a = FermionOperator('0 1^') + FermionOperator('2^ 3', {"a": -3})
    assert a == sum(list(a))
    b = FermionOperator('0 1^')
    b_exp = [FermionOperator('1^'), FermionOperator('0')]
    for idx, o in enumerate(b.singlet()):
        assert o == b_exp[idx]
    assert b.singlet_coeff() == -1
    assert b.is_singlet


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dumps_and_loads():
    """
    Description: Test fermion operator dumps to json and json loads to fermion operator
    Expectation:
    """
    f = FermionOperator('0', 1 + 2j) + FermionOperator('0^', 'a')
    strings = f.dumps()
    obj = FermionOperator.loads(strings)
    assert obj == f


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.skipif(not _HAS_OPENFERMION or not _FORCE_TEST, reason='OpenFermion is not installed')
@pytest.mark.skipif(not _FORCE_TEST, reason='Set not force test')
def test_of_fermion_trans():
    """
    Description: Test transform fermion operator to openfermion back and force.
    Expectation: success.
    """
    ofo_ops = OFFermionOperator("1^ 0", 1)
    mq_ops = FermionOperator('1^ 0', 1)
    assert mq_ops.to_openfermion() == ofo_ops
    assert mq_ops == FermionOperator.from_openfermion(ofo_ops)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_pickle_fermion():
    """
    Description: Test pickle for fermion operator
    Expectation: success.
    """
    ops = FermionOperator('1^ 0')
    assert ops == pickle.loads(pickle.dumps(ops))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_relabel():
    """
    Description: Test relabel of fermion operator
    Expectation:
    """
    f_op = FermionOperator('3^ 2 1 0')
    f_op = f_op.relabel([1, 3, 0, 2])
    assert f_op == -FermionOperator('3 2^ 1 0')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_get_matrix():
    """
    Description: Test get matrix of fermion operator.
    Expectation: success.
    """
    a = FermionOperator('0', 'a')
    with pytest.raises(ValueError, match="Parameter a not in given parameter resolver."):
        a.matrix()
    assert np.allclose(a.matrix(pr={'a': 1}).toarray(), FermionOperator('0').matrix().toarray())
