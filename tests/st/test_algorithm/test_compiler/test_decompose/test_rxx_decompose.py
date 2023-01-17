<<<<<<<< HEAD:tests/st/test_algorithm/test_compiler/test_decompose/test_fixed_decompose/test_xx_decompose.py
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

# pylint: disable=invalid-name
'''test decompose rule'''
import numpy as np
import pytest

from mindquantum.algorithm.compiler.decompose import cxx_decompose, xx_decompose
from mindquantum.config import Context
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import XX


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_xx(dtype):
    """
    Description: Test xx decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    xx = XX(1).on([0, 1])
    for solution in xx_decompose(xx):
        circuit_equal_test(xx, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_cxx(dtype):
    """
    Description: Test cxx decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    cxx = XX(2).on([0, 1], [2, 3])
    for solution in cxx_decompose(cxx):
        circuit_equal_test(cxx, solution)
========
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

# pylint: disable=invalid-name
'''test decompose rule'''
import numpy as np
import pytest

from mindquantum.algorithm.compiler.decompose import crxx_decompose, rxx_decompose
from mindquantum.config import Context
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import Rxx


def circuit_equal_test(gate, decompose_circ):
    """
    require two circuits are equal.
    """
    orig_circ = Circuit() + gate
    assert np.allclose(orig_circ.matrix(), decompose_circ.matrix())


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_rxx(dtype):
    """
    Description: Test rxx decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    rxx = Rxx(1).on([0, 1])
    for solution in rxx_decompose(rxx):
        circuit_equal_test(rxx, solution)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', ['float', 'double'])
def test_crxx(dtype):
    """
    Description: Test crxx decompose
    Expectation: success
    """
    Context.set_dtype(dtype)
    cxx = Rxx(2).on([0, 1], [2, 3])
    for solution in crxx_decompose(cxx):
        circuit_equal_test(cxx, solution)
>>>>>>>> e4807eae (xx yy zz to rxx ryy rzz):tests/st/test_algorithm/test_compiler/test_decompose/test_rxx_decompose.py
