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
"""Test ParameterResolve."""
import pickle

import pytest

import mindquantum as mq
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.core.parameterresolver import PRGenerator


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_params_name_order():
    """
    Description: Test parameters name of ansatz and encoder parameters
    Expectation: success
    """
    pr = PR(dict(zip([str(i) for i in range(10)], range(10))))
    assert pr.params_name == pr.ansatz_parameters


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_parameter_resolve():
    """
    Description: Test parameter resolver
    Expectation:
    """
    pr = PR({'a': 1.0})
    pr['b'] = 2.0
    pr['c'] = 3.0
    pr['d'] = 4.0
    pr *= 2
    pr = pr * 2
    pr = 1 * pr
    pr_tmp = PR({'e': 5.0, 'f': 6.0})
    pr_tmp.no_grad()
    pr.update(pr_tmp)
    assert pr.params_name == ['a', 'b', 'c', 'd', 'e', 'f']
    assert list(pr.params_value) == [4.0, 8.0, 12.0, 16.0, 5.0, 6.0]
    pr.requires_grad_part('e')
    pr.no_grad_part('b')
    assert set(pr.requires_grad_parameters) == {'a', 'c', 'd', 'e'}
    assert set(pr.no_grad_parameters) == {'b', 'f'}
    pr.requires_grad()
    assert not pr.no_grad_parameters


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', [mq.complex128, mq.float64])
def test_parameter_resolve_dumps_and_loads(dtype):
    '''
    Description: Test pr dumps to json and json loads to pr
    Expectation:
    '''
    pr = PR({'a': 1, 'b': 2, 'c': 3, 'd': 4}, dtype=dtype)
    pr.no_grad_part('a', 'b')

    string = pr.dumps()
    obj = PR.loads(string)
    assert obj == pr


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', [mq.complex128, mq.complex64, mq.float32, mq.float64])
def test_parameter_resolve_combination(dtype):
    """
    Description: Test pr combination
    Expectation:
    """
    pr1 = PR({'a': 1}, dtype=dtype)
    pr2 = PR({'a': 2, 'b': 3}, dtype=dtype)
    assert pr1.combination(pr2) == 2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', [mq.complex128, mq.complex64, mq.float32, mq.float64])
def test_parameter_resolver_pickle(dtype):
    """
    Description: Test pickle
    Expectation: success
    """
    pr = PR({'a': 1.2}, dtype=dtype)
    assert pr == pickle.loads(pickle.dumps(pr))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_pr_generator():
    """
    Description: Test pr generator
    Expectation: success
    """
    pr_gen = PRGenerator(prefix='l')
    pr1 = pr_gen.new()
    pr2 = pr_gen.new(suffix='a')
    assert pr1 == PR('l_p0')
    assert pr2 == PR('l_p1_a')
    assert pr_gen.size() == 2
    pr_gen.reset()
    assert pr_gen.size() == 0
    assert pr_gen.new() == pr1
