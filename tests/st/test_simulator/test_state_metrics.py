# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Validation of state-comparison inputs."""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.simulator import Simulator, fidelity, inner_product

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


@pytest.mark.parametrize(
    'other,error,match',
    [
        pytest.param(lambda: Simulator('mqvector', 3), ValueError, 'same quantum state', id='width'),
        pytest.param(lambda: Simulator('mqmatrix', 2), ValueError, 'backend', id='backend'),
        pytest.param(lambda: Simulator('mqvector', 2, dtype=mq.complex64), TypeError, 'Data type', id='precision'),
    ],
)
def test_inner_product_validation(other, error, match):
    """
    Feature: Simulator inner product.
    Description: Compare simulators with incompatible width, backend or precision.
    Expectation: Each mismatch is rejected rather than silently converted.
    """
    with pytest.raises(error, match=match):
        inner_product(Simulator('mqvector', 2), other())


@pytest.mark.parametrize(
    'left,right,match',
    [
        pytest.param(np.ones(2), np.ones(4), 'size of two', id='size-mismatch'),
        pytest.param(np.ones(3), np.ones(3), 'power of 2', id='non-qubit-size'),
        pytest.param(np.ones(2), np.ones(2), 'normalized', id='vector-norm'),
        pytest.param([[0, 1], [0, 0]], [[0, 1], [0, 0]], 'hermitian', id='non-hermitian'),
        pytest.param(np.diag([-0.5, 1.5]), np.diag([-0.5, 1.5]), 'negative', id='negative-population'),
        pytest.param(np.diag([2, 0]), np.diag([2, 0]), 'trace', id='trace'),
        pytest.param(np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), 'vector or matrix', id='rank'),
        pytest.param(np.zeros((2, 3)), np.zeros((2, 3)), 'row of matrix', id='non-square'),
    ],
)
def test_fidelity_validation(left, right, match):
    """
    Feature: Fidelity input validation.
    Description: Compare states with invalid dimensions or normalization.
    Expectation: Invalid states raise ValueError with a diagnostic identifying the violated contract.
    """
    with pytest.raises(ValueError, match=match):
        fidelity(left, right)


def test_fidelity_list_conversion():
    """
    Feature: Fidelity array-like inputs.
    Description: Compare distinct normalized states supplied as Python lists.
    Expectation: Lists are converted and the squared overlap is returned.
    """
    assert fidelity([1.0, 0.0], [0.6, 0.8]) == pytest.approx(0.36)
