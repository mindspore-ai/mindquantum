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

from mindquantum import ParameterResolver as PR


def test_parameter_resolve():
    """test parameter resolver."""
    pr = PR({'a': 1.0})
    pr['b'] = 2.0
    pr[['c', 'd']] = [3.0, 4.0]
    pr *= 2
    pr = pr * 2
    pr = 1 * pr
    pr_tmp = PR({'e': 5.0, 'f': 6.0})
    pr_tmp.no_grad()
    pr.update(pr_tmp)
    assert pr.para_name == ['a', 'b', 'c', 'd', 'e', 'f']
    assert pr.para_value == [4.0, 8.0, 12.0, 16.0, 5.0, 6.0]
    pr.requires_grad_part('e')
    pr.no_grad_part('b')
    assert pr.requires_grad_parameters == {'a', 'c', 'd', 'e'}
    assert pr.no_grad_parameters == {'b', 'f'}
    pr.requires_grad()
    assert not pr.no_grad_parameters
    mindspore_data = pr.mindspore_data()
    assert 'gate_params_names' in mindspore_data
    assert 'gate_coeff' in mindspore_data
    assert 'gate_requires_grad' in mindspore_data
    assert sum(mindspore_data['gate_coeff']) == 51.0
    assert sum(mindspore_data['gate_requires_grad']) == 6
    assert ''.join(mindspore_data['gate_params_names']) == 'abcdef'


def test_parameter_resolve_combination():
    pr1 = PR({'a': 1})
    pr2 = PR({'a': 2, 'b': 3})
    assert pr1.combination(pr2) == 2
